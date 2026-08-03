"""
Persistent expert execution cache for Μῆτις MoE
================================================
Keeps active experts' stacked + dtype-cast weight tensors resident in GPU
memory across forwards, eliminating the per-forward ``torch.stack(views).to(dtype)``
re-materialisation that the grouped engine performs for every group of every
MoE layer.

Design
------
- **Entry key**: ``(sorted_group_tuple, dtype_str, requires_grad)`` —
  ``requires_grad`` is derived from ``torch.is_grad_enabled()`` and whether any
  source view requires grad, so inference (no-grad) and training (grad) builds
  never collide.
- **Staleness detection**: each entry stores a *signature*
  ``tuple((data_ptr, _version, shape) for t in sources)``.  On lookup the
  current signature is recomputed; a mismatch → stale → rebuild.  Shape is
  included to catch same-address, same-version, different-shaped aliases.
- **Framework invalidation is authoritative**: the signature is *best-effort*
  automatic detection.  It reliably catches in-place ops that bump
  ``_version`` (``copy_``, non-fused/foreach optimisers) and storage
  replacement that changes ``data_ptr`` (``load_state_dict(assign=True)``,
  EMA apply/restore, ``param.data = X``).  Three mutation patterns it
  **cannot** observe: fused CUDA optimisers (mutate storage in place, no
  ``_version`` bump, no ``data_ptr`` change), ``param.data = X`` when the
  CUDA caching allocator reuses the same block address, and in-place ops
  via the ``.data`` alias (``param.data.copy_(...)``, ``param.data.add_(...)``)
  which use a separate version counter from ``param``.  For all three,
  correctness depends on the framework's own training loop
  (``training.py``, ``cuda_graphs.py``) calling ``invalidate()`` after
  every ``optimizer.step()`` / ``scaler.step()`` — custom training loops
  must call ``model.invalidate_moe_caches()`` after each weight update.
- **Oversize-entry protection**: when ``byte_capacity > 0`` and a newly
  built entry would exceed the budget on its own, the entry is returned
  without being cached — preventing a single large group from evicting
  every other entry and collapsing the cache to pure overhead.
- **Thread safety**: ``get_or_build``, ``invalidate``, and ``reset`` are
  protected by a ``threading.Lock`` so concurrent inference threads
  (e.g. the server's streaming path) do not corrupt the LRU or byte
  accounting.  The expensive ``build()`` callable executes outside the lock.
- **Honest bandwidth accounting**: in addition to stack+cast remat bytes
  (``bytes_saved`` / ``bytes_built``), the cache tracks per-forward GEMM
  weight reads (``bytes_read`` — the cast tensors consumed by
  ``grouped_gemm`` and ``grouped_output_projection`` every forward).
  ``bandwidth_reduction_pct`` reports total MoE weight-traffic reduction
  (including reads); ``stackcast_avoided_pct`` reports the remat-only
  metric.
- **Byte accounting** (exact, from tensor metadata): on a rebuild the
  stack+cast pipeline moves ``remat = 3·src_bytes + 2·cast_bytes`` device
  bytes (stack reads sources + writes stacked fp32; cast reads stacked +
  writes cast).  On a hit these bytes are added to ``bytes_saved``; on a
  miss to ``bytes_built``.  ``bytes_read = resident`` per lookup (the cast
  tensors read by the GEMM).

``ExpertCache`` is a plain Python class — not an ``nn.Module`` — so it is
never included in ``state_dict`` and never persists across checkpoint
round-trips.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import torch

__all__ = [
    "ExpertCache",
    "expert_cache_hit_rate",
    "expert_cache_bandwidth_reduction",
]


# ──────────────────────────────────────────────────────────────────────────────
# Byte accounting helpers
# ──────────────────────────────────────────────────────────────────────────────

def _signature(sources: list[torch.Tensor]) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Build a staleness signature from a list of source weight views.

    Uses ``data_ptr()`` (stable across re-created views of the same param,
    changes when ``load_state_dict``/``assign=True`` replaces the storage),
    ``_version`` (bumped by ``copy_()`` and non-fused optimisers), and
    ``shape`` (catches same-address, same-version, different-shaped aliases).
    """
    return tuple((t.data_ptr(), t._version, tuple(t.shape)) for t in sources)


def _remat_bytes(
    sources: list[torch.Tensor], cast_tensor: torch.Tensor
) -> int:
    """Exact device bytes moved by the stack + cast pipeline for one group.

    ``sources`` are the fp32 master weight views (w1 + w2 for the group's
    experts).  ``cast_tensor`` is one of the two outputs (both have the same
    element count and dtype, so either works).

    Traffic model per tensor:
      stack:  read sources (src_bytes) + write stacked fp32 (src_bytes)
      cast:   read stacked fp32 (src_bytes) + write cast (cast_bytes)
    Total: 3·src_bytes + cast_bytes
    For *both* w1 and w2 combined: 3·src_bytes + 2·cast_bytes
    (w1/w2 sources are the combined list; w1/w2 cast tensors are symmetric).
    """
    src_bytes = sum(t.numel() * t.element_size() for t in sources)
    cast_bytes = cast_tensor.numel() * cast_tensor.element_size()
    return 3 * src_bytes + 2 * cast_bytes


def _gmm_read_bytes(w1: torch.Tensor, w2: torch.Tensor) -> int:
    """Per-forward GEMM weight-read bytes for a group.

    The cached ``w1`` and ``w2`` tensors are read by ``grouped_gemm`` and
    ``grouped_output_projection`` on every forward — this traffic is
    independent of cache hit/miss and is tracked separately from the
    stack+cast remat bytes.
    """
    return w1.numel() * w1.element_size() + w2.numel() * w2.element_size()


# ──────────────────────────────────────────────────────────────────────────────
# Cache entry (internal)
# ──────────────────────────────────────────────────────────────────────────────

class _Entry:
    """One cached ``(w1_group, w2_group)`` pair with bookkeeping."""

    __slots__ = ("sig", "w1", "w2", "remat", "resident", "side_event", "prefetched")

    def __init__(
        self,
        sig: tuple,
        w1: torch.Tensor,
        w2: torch.Tensor,
        remat: int,
        resident: int,
        side_event: object | None = None,
        prefetched: bool = False,
    ):
        self.sig = sig
        self.w1 = w1
        self.w2 = w2
        self.remat = remat      # device bytes avoided on a hit
        self.resident = resident  # bytes occupied in this cache entry
        self.side_event = side_event  # cuda.Event when built on a prefetch stream
        self.prefetched = prefetched  # inserted by prefetch() (speculative)


# ──────────────────────────────────────────────────────────────────────────────
# ExpertCache
# ──────────────────────────────────────────────────────────────────────────────

class ExpertCache:
    """Bounded LRU cache of stacked + dtype-cast expert group weight tensors.

    Args:
        entry_capacity: maximum number of ``(group, dtype, requires_grad)``
            entries (``0`` = disabled).
        byte_capacity: optional device-byte budget; entries are evicted when
            the resident total exceeds this (``0`` = unbounded by bytes).

    Thread safety: ``get_or_build``, ``invalidate``, and ``reset`` are
    protected by a lock.  The expensive ``build()`` callable is invoked
    *outside* the lock so concurrent misses for different groups do not
    serialize their rebuilds.

    Usage::

        cache = ExpertCache(entry_capacity=64)

        # Inside forward_grouped, per group:
        w1, w2 = cache.get_or_build(
            group=(0, 2, 5),        # sorted expert ids
            dtype=torch.bfloat16,   # target cast dtype
            sources=[w1_views[0], w1_views[2], w1_views[5],
                     w2_views[0], w2_views[2], w2_views[5]],
            build=lambda: (
                torch.stack([w1_views[0], w1_views[2], w1_views[5]]).to(torch.bfloat16),
                torch.stack([w2_views[0], w2_views[2], w2_views[5]]).to(torch.bfloat16),
            ),
        )
    """

    def __init__(
        self,
        entry_capacity: int = 64,
        byte_capacity: int = 0,
    ):
        self.entry_capacity: int = max(0, int(entry_capacity))
        self.byte_capacity: int = max(0, int(byte_capacity))
        self.enabled: bool = self.entry_capacity > 0

        # Thread-safety lock for the entry store and counters.
        self._lock: threading.Lock = threading.Lock()

        # LRU ordered dict: key -> _Entry
        self._entries: OrderedDict = OrderedDict()

        # Cumulative statistics (never reset by invalidate())
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.bytes_saved: int = 0   # bytes avoided on hits (stack+cast)
        self.bytes_built: int = 0   # bytes moved on misses (rebuilds)
        self.bytes_read: int = 0    # per-forward GEMM weight reads
        self.oversized_skips: int = 0  # entries too large to cache
        self.prefetched: int = 0    # speculative prefetch builds issued
        self.prefetch_useful: int = 0  # prefetched entries later served as hits

        # Current resident cache size in device bytes
        self._resident: int = 0

    # ── public API ────────────────────────────────────────────────────────

    def get_or_build(
        self,
        group,
        dtype: torch.dtype,
        sources: list[torch.Tensor],
        build,
    ):
        """Look up a cached ``(w1_group, w2_group)`` or build and insert it.

        The ``build()`` callable is invoked *outside* the internal lock, so
        concurrent misses for different groups rebuild in parallel.

        Args:
            group: sorted expert ids (iterable of int).
            dtype: target tensor dtype for the cast.
            sources: ``w1_views + w2_views`` for the group's experts (used
                only for staleness signature and byte accounting; the view
                objects themselves are not stored by the cache).  Note that a
                grad-built cached tensor (``requires_grad``) retains a
                ``grad_fn`` chain that references the source params — that is
                required for gradient flow during training, and ``_resident``
                counts the cache's own cast-tensor allocations (the params are
                owned by the model either way).
            build: zero-arg callable returning ``(w1_group, w2_group)``.

        Returns:
            ``(w1_group, w2_group)`` — cached or freshly built.
        """
        if not self.enabled:
            self.misses += 1
            w1, w2 = build()
            remat = _remat_bytes(sources, w1)
            self.bytes_built += remat
            self.bytes_read += _gmm_read_bytes(w1, w2)
            return w1, w2

        sig = _signature(sources)
        key = (
            tuple(int(e) for e in group),
            str(dtype),
            bool(
                torch.is_grad_enabled()
                and any(t.requires_grad for t in sources)
            ),
        )

        # Fast path: check the cache under the lock.
        w1 = w2 = side_event = None
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.sig == sig:
                # ── hit ──
                self.hits += 1
                self.bytes_saved += entry.remat
                self.bytes_read += entry.resident  # GEMM reads this forward
                self._entries.move_to_end(key)
                w1, w2 = entry.w1, entry.w2
                side_event = entry.side_event
                if entry.prefetched:
                    # A prefetch built this entry — count it as useful. If the
                    # build ran on a side stream, the first consumer waits for
                    # it below (query() short-circuits once it's complete).
                    self.prefetch_useful += 1

        # Sync outside the lock: never hold the store lock during a device
        # query/sync.  query() is cheap once the prefetch build has completed
        # (the normal case — it ran during the *previous* layer's compute).
        if side_event is not None and not side_event.query():
            torch.cuda.current_stream().wait_event(side_event)
        if w1 is not None:
            return w1, w2

        # ── miss ── build outside the lock so concurrent groups rebuild in
        # parallel (the lock only guards the entry store and counters).
        self.misses += 1
        w1, w2 = build()
        remat = _remat_bytes(sources, w1)
        read = _gmm_read_bytes(w1, w2)
        resident = w1.numel() * w1.element_size() + w2.numel() * w2.element_size()

        with self._lock:
            # Double-check: another thread may have inserted while we were
            # building.  If the signature now matches, use the fresh copy.
            entry = self._entries.get(key)
            if entry is not None and entry.sig == sig:
                self.hits += 1
                self.bytes_saved += entry.remat
                self.bytes_read += read
                self._entries.move_to_end(key)
                w1, w2 = entry.w1, entry.w2
                side_event = entry.side_event
                if entry.prefetched:
                    self.prefetch_useful += 1
            else:
                # Insert the freshly built entry (still under the lock). The
                # fresh build ran on the caller's stream, so no side_event.
                self.bytes_built += remat
                self.bytes_read += read
                if entry is not None:
                    # Stale entry replaced — update resident accounting.
                    self._resident -= entry.resident
                if self.byte_capacity > 0 and resident > self.byte_capacity:
                    self.oversized_skips += 1
                else:
                    self._entries[key] = _Entry(sig, w1, w2, remat, resident)
                    self._resident += resident
                    self._entries.move_to_end(key)
                    self._evict()
                side_event = None

        if side_event is not None and not side_event.query():
            torch.cuda.current_stream().wait_event(side_event)
        return w1, w2

    def prefetch(
        self,
        group,
        dtype: torch.dtype,
        sources: list[torch.Tensor],
        build,
        stream=None,
    ) -> None:
        """Speculatively build a group's stacked+cast weights on a side stream.

        Layer prefetching calls this during the *previous* layer's compute so
        the next layer's ``get_or_build`` finds the group resident (a hit) and
        never stalls on the synchronous stack+cast. The build is the identical
        stack+cast the miss path would run, and the entry is still governed by
        the staleness signature — a later real lookup rebuilds if the weights
        changed. Speculative work is tracked separately (``prefetched`` /
        ``prefetch_useful``); it never touches ``hits``/``misses``/``bytes_built``.

        Args:
            group: sorted expert ids (ascending — matches ``get_or_build`` keys).
            dtype: target tensor dtype for the cast.
            sources: ``w1_views + w2_views`` for the group (staleness signature).
            build: zero-arg callable returning ``(w1_group, w2_group)``.
            stream: CUDA stream to run the build on (the prefetch stream);
                ``None`` → the current stream (CPU or synchronous warm-up).
        """
        if not self.enabled:
            return
        sig = _signature(sources)
        key = (
            tuple(int(e) for e in group),
            str(dtype),
            bool(
                torch.is_grad_enabled()
                and any(t.requires_grad for t in sources)
            ),
        )

        # Already resident and fresh?  Nothing to do (the real lookup will hit).
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.sig == sig:
                self._entries.move_to_end(key)
                return

        self.prefetched += 1
        if stream is not None and torch.cuda.is_available():
            with torch.cuda.stream(stream):
                w1, w2 = build()
                side_event = torch.cuda.Event()
                side_event.record(stream)
        else:
            w1, w2 = build()
            side_event = None

        remat = _remat_bytes(sources, w1)
        resident = w1.numel() * w1.element_size() + w2.numel() * w2.element_size()

        with self._lock:
            # Another thread may have inserted while we built — keep the
            # freshest entry (signature is authoritative either way).
            entry = self._entries.get(key)
            if entry is not None and entry.sig == sig:
                return
            if entry is not None:
                self._resident -= entry.resident
            # Oversize-entry protection (mirrors get_or_build).
            if self.byte_capacity > 0 and resident > self.byte_capacity:
                self.oversized_skips += 1
                return
            self._entries[key] = _Entry(sig, w1, w2, remat, resident,
                                        side_event, prefetched=True)
            self._resident += resident
            self._entries.move_to_end(key)
            self._evict()

    def invalidate(self) -> None:
        """Drop all cached entries (keep running statistics).

        Called by the framework after weight-mutating operations (optimizer
        steps, ``load_state_dict``, EMA apply/restore).
        """
        with self._lock:
            self._entries.clear()
            self._resident = 0

    def reset(self) -> None:
        """Drop all entries **and** zero all statistics.

        Used on device / dtype changes (``model.to(...)``).
        """
        with self._lock:
            self._entries.clear()
            self._resident = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.bytes_saved = 0
            self.bytes_built = 0
            self.bytes_read = 0
            self.oversized_skips = 0
            self.prefetched = 0
            self.prefetch_useful = 0

    def stats(self) -> dict:
        """Return a snapshot of cache statistics."""
        total = self.hits + self.misses
        remat_total = self.bytes_saved + self.bytes_built
        return {
            "enabled": self.enabled,
            "entry_capacity": self.entry_capacity,
            "byte_capacity": self.byte_capacity,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "evictions": self.evictions,
            "oversized_skips": self.oversized_skips,
            "prefetched": self.prefetched,
            "prefetch_useful": self.prefetch_useful,
            "prefetch_accuracy": (
                self.prefetch_useful / self.prefetched if self.prefetched else 0.0
            ),
            "bytes_saved": self.bytes_saved,
            "bytes_built": self.bytes_built,
            "bytes_read": self.bytes_read,
            "bandwidth_reduction_pct": (
                100.0 * self.bytes_saved / (self.bytes_saved + self.bytes_built + self.bytes_read)
                if (self.bytes_saved + self.bytes_built + self.bytes_read) else 0.0
            ),
            "stackcast_avoided_pct": (
                100.0 * self.bytes_saved / remat_total
                if remat_total else 0.0
            ),
            "resident_bytes": self._resident,
        }

    # ── convenience functions (module-level helpers re-exported) ───────────

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"ExpertCache(entries={len(self._entries)}/{self.entry_capacity}, "
            f"hit_rate={stats['hit_rate']:.1%}, "
            f"bw_reduction={stats['bandwidth_reduction_pct']:.1f}%, "
            f"resident={self._resident:,} B)"
        )

    # ── internal ──────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Evict oldest entries to respect capacity constraints."""
        while len(self._entries) > self.entry_capacity:
            _key, entry = self._entries.popitem(last=False)
            self._resident -= entry.resident
            self.evictions += 1

        if self.byte_capacity > 0:
            while self._resident > self.byte_capacity and self._entries:
                _key, entry = self._entries.popitem(last=False)
                self._resident -= entry.resident
                self.evictions += 1


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience helpers
# ──────────────────────────────────────────────────────────────────────────────


def expert_cache_hit_rate(cache: ExpertCache | None) -> float:
    """Return the hit rate of a cache, or ``0.0`` if the cache is ``None``."""
    if cache is None:
        return 0.0
    return cache.stats()["hit_rate"]


def expert_cache_bandwidth_reduction(cache: ExpertCache | None) -> float:
    """Return total MoE weight-traffic reduction (0-100 %) or ``0.0``.

    Includes both stack+cast remat avoided and per-forward GEMM weight
    reads in the denominator, giving an honest measure of the cache's
    contribution to reducing total memory traffic.  For the remat-only
    metric, use ``cache.stats()["stackcast_avoided_pct"]``.
    """
    if cache is None:
        return 0.0
    return cache.stats()["bandwidth_reduction_pct"]
