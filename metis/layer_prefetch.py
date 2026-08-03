"""
Layer-level expert prefetching for Μῆτις MoE
=============================================
While layer ``N`` computes, speculatively build layer ``N+1``'s expert-cache
entries on a dedicated CUDA stream, so layer ``N+1``'s ``get_or_build`` calls
hit and never stall on the synchronous ``torch.stack(views).to(dtype)`` that a
cache miss performs on the compute stream.

Prediction uses **temporal locality**: every MoE layer records the expert
groups it routed to; the next forward prefetches each layer's groups during
the *preceding* layer's compute.  Routing is stable across steps (very stable
in KV-cache decode), so most prefetches become hits.  A prediction that misses
is harmless — the ``ExpertCache`` signature check still rebuilds on demand and
the speculative entry is evicted by LRU.

Correctness
-----------
- The prefetch invokes the *identical* stack+cast the miss path would run, on
  the same source views at the current weights → deterministic output.
- The cache's ``(data_ptr, _version, shape)`` staleness signature is still
  authoritative on every real lookup; a weight change between prefetch and use
  forces a rebuild.
- Stream ordering: prefetched entries record a completion event; the compute
  stream ``wait_event``s a side-built entry on its first hit (a no-op once the
  build finished during the previous layer).
- CPU / no-CUDA: the prefetcher is a synchronous speculative warm-up (still
  improves hit rate under cache pressure; there is no stream overlap).

The prefetcher is plain Python (not an ``nn.Module``) — it holds no parameters
and never appears in ``state_dict``.
"""

from __future__ import annotations

import torch

__all__ = ["LayerExpertPrefetcher"]


class LayerExpertPrefetcher:
    """Speculatively warm the expert cache of the *next* MoE layer.

    The model's forward loop calls :meth:`prefetch_next` before each layer;
    each MoE layer reports its routed groups via :meth:`record`.  Prefetches
    for layer ``i+1`` use layer ``i+1``'s own routing from the previous
    forward (temporal locality).
    """

    def __init__(self, layers, stream: torch.cuda.Stream | None = None) -> None:
        # Aligned to layer index: the MoE ffn, or None for non-MoE layers.
        self._moes = [
            layer.ffn if hasattr(getattr(layer, "ffn", None), "experts") else None
            for layer in layers
        ]
        # layer_idx -> list of (group_tuple, eff_dtype) from the last forward.
        self._prev_groups: dict[int, list] = {}
        # layer_idx -> (w1_views, w2_views, signature) — the transposed expert
        # views are rebuilt only when the underlying weights change (the
        # signature is the same (data_ptr, _version, shape) check the cache
        # uses). Avoids re-creating N view objects every prefetch call.
        self._views_cache: dict[int, tuple] = {}
        if stream is not None:
            self._stream = stream
        elif torch.cuda.is_available():
            self._stream = torch.cuda.Stream(device="cuda")
        else:
            self._stream = None
        # Prefetcher-level stats (builds/useful come from each ExpertCache).
        self.prefetch_calls = 0   # times prefetch_next issued work
        self.predictions = 0      # predicted (group, dtype) pairs consulted

    def record(self, layer_idx: int, groups, eff_dtype: torch.dtype) -> None:
        """Store the routed expert groups for a layer (for the next forward)."""
        self._prev_groups[layer_idx] = [
            (tuple(int(e) for e in g), eff_dtype) for g in groups
        ]

    def prefetch_next(self, layer_idx: int) -> None:
        """Issue speculative cache builds for layer ``layer_idx + 1``.

        The prediction is that layer ``i+1`` routes to the same expert groups
        as its previous forward.  Builds run on the prefetch stream, overlapped
        with layer ``layer_idx``'s compute.
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self._moes) or self._moes[next_idx] is None:
            return
        preds = self._prev_groups.get(next_idx)
        if not preds:
            return  # cold start — no routing history yet
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return  # never run stream work inside a graph capture

        moe = self._moes[next_idx]
        cache = getattr(moe, "_cache", None)
        if cache is None:
            return  # cache disabled (e.g. torch.compile) — nothing to warm

        cached = self._views_cache.get(next_idx)
        if cached is not None:
            w1_views, w2_views, sig = cached
            cur_sig = tuple(
                (v.data_ptr(), v._version, v.shape) for v in (w1_views + w2_views)
            )
            if cur_sig != sig:
                cached = None  # weights changed (optimizer step) — refresh
        if cached is None:
            w1_views = [e[0].weight.t() for e in moe.experts]
            w2_views = [e[2].weight.t() for e in moe.experts]
            sig = tuple(
                (v.data_ptr(), v._version, v.shape) for v in (w1_views + w2_views)
            )
            self._views_cache[next_idx] = (w1_views, w2_views, sig)
        self.prefetch_calls += 1
        self.predictions += len(preds)
        for group, dtype in preds:
            sources = [w1_views[i] for i in group] + [w2_views[i] for i in group]
            cache.prefetch(
                group, dtype, sources,
                build=lambda g=group, d=dtype: (
                    torch.stack([w1_views[i] for i in g]).to(d),
                    torch.stack([w2_views[i] for i in g]).to(d),
                ),
                stream=self._stream,
            )

    def clear(self) -> None:
        """Drop all routing records (no stale predictions after a weight update)."""
        self._prev_groups.clear()

    def stats(self) -> dict:
        return {
            "layers": len(self._moes),
            "recorded_layers": sum(1 for v in self._prev_groups.values() if v),
            "prefetch_calls": self.prefetch_calls,
            "predictions": self.predictions,
        }
