"""
Μῆτις (Metis) — KV cache subsystem
====================================
Optional, flag-gated KV cache engines that replace the legacy *growable*
per-layer ``(K, V)`` tuple cache during inference while preserving the public
API contract: ``model.forward(idx, ..., kv_cache=None) -> (logits, loss,
new_kv_cache)`` where ``new_kv_cache`` is passed back verbatim on the next
call. The cache object is opaque to callers — the model, ``generate_text``,
the execution scheduler and the web/server front-ends all just round-trip it.

Backends (``ModelConfig.kv_backend``):

  ``"default"``   — the legacy growable cache (``torch.cat`` per step). This
                    module does **not** implement it; it remains in
                    ``metis/model.py`` untouched so existing behavior is
                    byte-identical. Kept here only as the analytical baseline.

  ``"static"``    — preallocated contiguous ``(B, n_kv, max_seq_len, head_dim)``
                    buffers per layer, written in place (``copy_`` into a
                    slice) with a length tracker. Eliminates the per-step
                    ``torch.cat`` (O(T) allocation+copy per step → O(T²) total),
                    keeps cache memory flat, and is **bit-identical** to the
                    default backend (verified in
                    ``benchmarks/verify_kv_parity.py``). ``kv_cache_dtype``
                    selects the storage element type (``"auto"`` = compute
                    dtype; ``"fp16"`` / ``"bf16"`` halve memory).

  ``"quantized"`` — the static layout plus an int8 cache. K and V are
                    quantized on write with per-token symmetric scales
                    (one scale per ``(B, n_kv, T)``), dequantized on read.
                    Roughly a 4x cache-memory cut over fp32 with measured,
                    bounded error (see ``docs/kv_cache.md``).

The MLA backend (``kv_backend="mla"``) is an *architecture* change implemented
in ``metis/mla.py``; it stores a compressed latent per token and does not use
this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Quantization helpers (per-token symmetric int8)
# ──────────────────────────────────────────────────────────────────────────────

_QMIN, _QMAX = -127, 127


def quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 quantization with a per-token scale.

    Args:
        x: ``(B, n_kv, T, D)`` in any floating dtype.
    Returns:
        (``q`` int8 ``(B, n_kv, T, D)``, ``scale`` ``(B, n_kv, T, 1)``).
        ``x ≈ q * scale`` (with ``scale = amax / 127``). A zero row maps to
        scale 1 (guard).
    """
    amax = x.abs().amax(dim=-1, keepdim=True)          # (B, n_kv, T, 1)
    scale = amax / _QMAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(x / scale), _QMIN, _QMAX).to(torch.int8)
    return q, scale


def dequantize_per_token(q: torch.Tensor, scale: torch.Tensor,
                         dtype: torch.dtype) -> torch.Tensor:
    """Inverse of :func:`quantize_per_token`, cast back to ``dtype``.

    Symmetric int8: ``scale = amax / 127`` so ``x ≈ q * scale`` (q is in
    ``[-127, 127]``). Zero-rows were stored with ``scale = 1`` at quantize
    time, so a zero payload dequantizes to zero.
    """
    return (q.float() * scale).to(dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Analytical memory model (used by benchmarks / docs without allocating)
# ──────────────────────────────────────────────────────────────────────────────

def _elem_bytes(dtype: torch.dtype | str) -> int:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype) if dtype != "auto" else torch.float32
    return torch.tensor([], dtype=dtype).element_size()


def cache_memory_bytes(
    backend: str,
    *,
    B: int,
    n_kv_heads: int,
    head_dim: int,
    T: int,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
    mla_kv_latent_dim: int = 0,
    mla_rope_head_dim: int = 0,
    n_heads: int = 0,
) -> int:
    """Bytes one layer's KV cache occupies at context length ``T``.

    ``"default"`` reports exactly what it holds at ``T`` (it reallocates to
    ``T`` each step). ``"static"`` reports its preallocated ``max_seq_len``
    footprint (constant — the price of zero per-step allocation). ``"quantized"``
    reports int8 K/V + per-token scales at ``T`` (flat, like static). ``"mla"``
    reports latent + RoPE keys at ``T``.
    """
    if backend == "mla":
        c_d = mla_kv_latent_dim or 0
        rope = mla_rope_head_dim or 0
        return B * T * (c_d + n_heads * rope) * _elem_bytes(dtype)
    if backend == "quantized":
        # int8 K + int8 V + two per-token scales (one float per (B, n_kv, T)).
        per = B * n_kv_heads * T
        return 2 * per * head_dim + 2 * per * _elem_bytes(dtype)
    if backend == "static":
        e = _elem_bytes(dtype)
        return 2 * B * n_kv_heads * max_seq_len * head_dim * e
    # default (growable)
    e = _elem_bytes(dtype)
    return 2 * B * n_kv_heads * T * head_dim * e


def kv_cache_ratio(
    backend: str, *, T: int, max_seq_len: int, **kw
) -> float:
    """Memory ratio of ``backend`` at length ``T`` vs the default cache."""
    default = cache_memory_bytes("default", T=T, max_seq_len=max_seq_len, **kw)
    other = cache_memory_bytes(backend, T=T, max_seq_len=max_seq_len, **kw)
    return default / max(other, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Per-layer cache
# ──────────────────────────────────────────────────────────────────────────────

class LayerKV:
    """One layer's cache: static preallocated buffers or quantized int8.

    Allocated lazily on the first :meth:`append` (which fixes ``B``, device,
    storage dtype and head count from the incoming K/V), so the container can
    be created before any forward and no buffers are wasted if a configured
    backend is never exercised (e.g. training forwards under a ``"static"``
    config pass ``kv_cache=None``).

    The object is intentionally small and index-compatible with a ``(K, V)``
    tuple: ``cache[0]`` / ``cache[1]`` return the *current* K / V views, so
    generic code that treated the old cache as a tuple still works.
    """

    __slots__ = (
        "backend", "config", "length",
        "_k", "_v", "_k_scale", "_v_scale", "_compute_dtype",
        "_B", "_n_kv", "_D", "_max_len", "_store_dtype",
    )

    def __init__(self, backend: str, config):
        assert backend in ("static", "quantized"), backend
        self.backend = backend
        self.config = config
        self.length = 0
        self._k = self._v = None
        self._k_scale = self._v_scale = None
        self._compute_dtype = None
        self._B = self._n_kv = self._D = self._max_len = None
        self._store_dtype = None

    # ── introspection ────────────────────────────────────────────────────

    @property
    def cached_len(self) -> int:
        return self.length

    def __len__(self) -> int:
        return 2  # tuple-compatible (K, V)

    def __getitem__(self, i: int) -> torch.Tensor:
        if not 0 <= i <= 1:
            raise IndexError(f"LayerKV index {i} out of range")
        k, v = self.keys_values()
        return (k, v)[i]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"LayerKV({self.backend}, len={self.length}/"
                f"{self._max_len}, B={self._B}, kv={self._n_kv}, D={self._D})")

    # ── write path ───────────────────────────────────────────────────────

    def _ensure(self, k: torch.Tensor) -> None:
        B, n_kv, T, D = k.shape
        self._max_len = self.config.max_seq_len
        store = self.config.kv_cache_dtype
        self._store_dtype = (
            k.dtype if store == "auto" else getattr(torch, store)
        )
        self._compute_dtype = k.dtype
        self._B, self._n_kv, self._D = B, n_kv, D
        if self.backend == "static":
            self._k = torch.empty(B, n_kv, self._max_len, D,
                                  device=k.device, dtype=self._store_dtype)
            self._v = torch.empty(B, n_kv, self._max_len, D,
                                  device=k.device, dtype=self._store_dtype)
        else:  # quantized — int8 payload + fp32-compute per-token scales
            self._k = torch.empty(B, n_kv, self._max_len, D,
                                  device=k.device, dtype=torch.int8)
            self._v = torch.empty(B, n_kv, self._max_len, D,
                                  device=k.device, dtype=torch.int8)
            self._k_scale = torch.empty(B, n_kv, self._max_len, 1,
                                        device=k.device, dtype=self._compute_dtype)
            self._v_scale = torch.empty(B, n_kv, self._max_len, 1,
                                        device=k.device, dtype=self._compute_dtype)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store a new chunk ``(B, n_kv, T, D)`` at the end of the cache."""
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} vs {v.shape}")
        if self.length == 0:
            self._ensure(k)
        T = k.size(2)
        if self.length + T > self._max_len:
            raise RuntimeError(
                f"KV cache overflow: {self.length} + {T} > max_seq_len "
                f"{self._max_len}. Call reset() (sliding window) first — "
                f"generate_text does this automatically."
            )
        s = self.length
        if self.backend == "static":
            self._k[:, :, s: s + T] = k.to(self._store_dtype)
            self._v[:, :, s: s + T] = v.to(self._store_dtype)
        else:
            kq, ks = quantize_per_token(k)
            vq, vs = quantize_per_token(v)
            self._k[:, :, s: s + T] = kq
            self._v[:, :, s: s + T] = vq
            self._k_scale[:, :, s: s + T] = ks.to(self._compute_dtype)
            self._v_scale[:, :, s: s + T] = vs.to(self._compute_dtype)
        self.length += T

    # ── read path ────────────────────────────────────────────────────────

    def keys_values(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current K / V views (``(B, n_kv, T_cur, D)``) for attention.

        Static returns zero-copy slices. Quantized dequantizes on read (the
        price of compression — the dequant cost is measured in the benchmark).
        """
        if self.length == 0:
            raise RuntimeError("LayerKV.keys_values() called on an empty cache")
        k = self._k[:, :, : self.length]
        v = self._v[:, :, : self.length]
        if self.backend == "quantized":
            k = dequantize_per_token(
                k, self._k_scale[:, :, : self.length], self._compute_dtype)
            v = dequantize_per_token(
                v, self._v_scale[:, :, : self.length], self._compute_dtype)
        elif self._store_dtype != self._compute_dtype:
            # fp16/bf16 static cache on an fp32 (or mixed) compute path.
            k = k.to(self._compute_dtype)
            v = v.to(self._compute_dtype)
        return k, v

    # ── lifecycle / memory ───────────────────────────────────────────────

    def reset(self) -> None:
        """Discard contents (sliding-window reset). Buffers are reused."""
        self.length = 0

    def allocated_bytes(self) -> int:
        """Resident bytes (constant for static; int8+scales for quantized)."""
        if self._k is None:
            return 0
        total = self._k.numel() * self._k.element_size()
        total += self._v.numel() * self._v.element_size()
        if self._k_scale is not None:
            total += self._k_scale.numel() * self._k_scale.element_size()
            total += self._v_scale.numel() * self._v_scale.element_size()
        return total

    def used_bytes(self) -> int:
        """Bytes of the live prefix (length rows only)."""
        if self._k is None:
            return 0
        live = self.length
        k_bytes = self._B * self._n_kv * live * self._D * self._k.element_size()
        v_bytes = self._B * self._n_kv * live * self._D * self._v.element_size()
        total = k_bytes + v_bytes
        if self._k_scale is not None:
            total += 2 * self._B * self._n_kv * live * self._k_scale.element_size()
        return total


# ──────────────────────────────────────────────────────────────────────────────
# Model-level cache container (list-like over layers)
# ──────────────────────────────────────────────────────────────────────────────

class KVCache:
    """Opaque, list-like container of :class:`LayerKV` — one per model layer.

    Indexing matches the legacy list-of-tuples contract (``kv_cache[i]`` is a
    layer cache), so the execution scheduler and generic callers that only
    index and round-trip keep working unchanged.
    """

    def __init__(self, backend: str, config, n_layers: int):
        assert backend in ("static", "quantized"), backend
        self.backend = backend
        self.config = config
        self.n_layers = n_layers
        self.layers = [LayerKV(backend, config) for _ in range(n_layers)]

    def __len__(self) -> int:
        return self.n_layers

    def __getitem__(self, i: int) -> LayerKV:
        return self.layers[i]

    @property
    def cached_len(self) -> int:
        """Live context length (layer 0 — all layers advance in lockstep)."""
        return self.layers[0].cached_len if self.n_layers else 0

    def reset(self) -> KVCache:
        for layer in self.layers:
            layer.reset()
        return self

    def allocated_bytes(self) -> int:
        return sum(layer.allocated_bytes() for layer in self.layers)

    def used_bytes(self) -> int:
        return sum(layer.used_bytes() for layer in self.layers)

    def memory_stats(self) -> dict:
        return {
            "backend": self.backend,
            "layers": self.n_layers,
            "cached_len": self.cached_len,
            "allocated_bytes": self.allocated_bytes(),
            "used_bytes": self.used_bytes(),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        s = self.memory_stats()
        return (f"KVCache(backend={s['backend']}, layers={s['layers']}, "
                f"len={s['cached_len']}, "
                f"used={s['used_bytes'] / 1e6:.2f}MB, "
                f"alloc={s['allocated_bytes'] / 1e6:.2f}MB)")

    @classmethod
    def from_legacy(cls, backend: str, config, legacy: list, n_layers: int) -> KVCache:
        """Seed a backend cache from a legacy ``[(K, V), ...]`` list.

        Lets callers pre-populate the cache (e.g. a warm chunked prefill) and
        hand the result to ``model.forward`` unchanged. Each layer's K/V is
        re-appended, so a legacy fp32 cache is compressed on ingestion.
        """
        cache = cls(backend, config, n_layers)
        for i, layer in enumerate(cache.layers):
            if i < len(legacy):
                k, v = legacy[i]
                layer.append(k, v)
        return cache


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers for cache consumers
# ──────────────────────────────────────────────────────────────────────────────

def cached_len_of(kv_cache) -> int:
    """Live context length of any supported cache form.

    Accepts ``None``, a legacy ``(K, V)`` tuple, a list of such tuples, a
    :class:`KVCache` (list of :class:`LayerKV`), or a bare :class:`LayerKV`.
    ``generate_text`` uses this for the sliding-window overflow check.
    """
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, LayerKV):
        return kv_cache.cached_len
    if isinstance(kv_cache, KVCache):
        return kv_cache.cached_len
    # list of per-layer caches (default backend) or a bare (K, V) tuple.
    first = kv_cache[0]
    if isinstance(first, LayerKV):
        return first.cached_len
    if hasattr(first, "cached_len"):       # MLALayerCache or future types
        return first.cached_len
    return first[0].size(2) if isinstance(first, (tuple, list)) else 0


def cached_bytes(kv_cache) -> int:
    """Resident cache bytes for any supported cache form."""
    if kv_cache is None:
        return 0
    if isinstance(kv_cache, (KVCache, LayerKV)):
        return kv_cache.allocated_bytes()
    total = 0
    for per_layer in kv_cache:
        for t in per_layer:
            total += t.numel() * t.element_size()
    return total


@dataclass(frozen=True)
class KVBackendInfo:
    """Static description of a backend — for ``metis info`` / docs."""

    name: str
    description: str
    per_token_elements: str
    bit_identical: bool
    needs_retrain: bool

    @classmethod
    def describe(cls, backend: str, config=None) -> KVBackendInfo:
        if backend == "mla":
            return cls(
                name="mla",
                description="Multi-head Latent Attention (architecture change)",
                per_token_elements="latent c_d + n_heads * rope_head_dim",
                bit_identical=False, needs_retrain=True,
            )
        if backend == "quantized":
            return cls(
                name="quantized",
                description="int8 per-token compressed cache (static layout)",
                per_token_elements="2 * n_kv_heads * (head_dim + 1) bytes",
                bit_identical=False, needs_retrain=False,
            )
        if backend == "static":
            return cls(
                name="static",
                description="preallocated contiguous buffers (flat memory)",
                per_token_elements="2 * n_kv_heads * head_dim * elem",
                bit_identical=True, needs_retrain=False,
            )
        return cls(
            name="default",
            description="legacy growable (K, V) tuple cache",
            per_token_elements="2 * n_kv_heads * head_dim * elem",
            bit_identical=True, needs_retrain=False,
        )
