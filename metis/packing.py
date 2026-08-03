"""
Μῆτις (Metis) — Dynamic Sequence Packing
===========================================
Packs variable-length training sequences (documents) into fixed-length batches
so that every token in a batch is real — no padding waste — while keeping
attention causal and document-segmented.

Two packing strategies:

* ``"stream"`` (1D / cross-boundary, the default) — documents are concatenated
  with ``<eos>`` separators into one contiguous token stream, then chopped into
  fixed-length sequences. A sequence boundary can fall mid-document and every
  token's next-token target stays valid (the corpus is contiguous), so there is
  **zero padding waste**. Within each packed sequence attention is
  block-diagonal: a token may only attend to earlier tokens of the same
  document.

* ``"bin"`` (whole-document first-fit-decreasing) — documents (``+<eos>``) are
  packed into bins of ``max_seq_len`` without ever splitting a document. Each
  bin is padded only up to ``max_seq_len``, minimising padding waste while
  keeping every document fully visible inside its own packed sequence.

Both strategies emit :class:`PackedBatch` objects carrying everything the model
needs: ``input_ids`` / ``labels``, a block-diagonal causal ``attention_mask``,
per-segment RoPE ``position_ids`` (positions reset at every document boundary),
and ``cu_seqlens`` (cumulative segment boundaries) for debugging / future
variable-length kernels.

The tokenizer interface is preserved: documents are ordinary token-ID lists
produced by ``BPETokenizer`` / ``CharTokenizer``; ``<eos>`` separates documents;
``<pad>`` fills ignored positions. The loss's ``ignore_index`` (pad, id 0)
matches the existing fixed-length pipeline exactly.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger("metis.packing")

STREAM = "stream"  # contiguous 1D packing — zero padding waste
BIN = "bin"        # whole-document first-fit-decreasing bin packing


# ──────────────────────────────────────────────────────────────────────────────
# Packing algorithms (pure, operate on token-ID lists)
# ──────────────────────────────────────────────────────────────────────────────

def _cu_seqlens_stream(chunk: np.ndarray, eos_id: int, real: int) -> list[int]:
    """Segment boundaries inside a stream-packed chunk.

    Position 0 always starts a segment. Every position whose *previous* token
    is ``<eos>`` starts a new segment (the ``<eos>`` itself ends the previous
    document's segment). The final boundary is ``real`` (chunk length).
    """
    starts = [0]
    for p in range(1, real):
        if int(chunk[p - 1]) == eos_id:
            starts.append(p)
    starts.append(real)
    return starts


def pack_stream(
    documents: Sequence[Sequence[int]],
    seq_len: int,
    eos_id: int | None,
    pad_id: int,
) -> tuple[list[np.ndarray], list[list[int]], list[np.ndarray], int]:
    """Pack documents into contiguous zero-padding chunks.

    Documents are concatenated with ``<eos>`` separators into one stream and
    chopped into consecutive ``seq_len`` chunks (the ragged tail is dropped,
    mirroring ``drop_last``). Labels are the *whole-stream* shift-by-one, so a
    token's next-token target stays valid across both document and chunk
    boundaries — nothing is lost.

    Args:
        documents: Sequence of token-ID lists.
        seq_len: Fixed length of every packed sequence.
        eos_id: Document separator token (``None`` = one continuous stream).
        pad_id: Fills the single corpus-end label that has no successor.

    Returns:
        ``(chunks, cu_seqlens, labels, n_pad)`` — each chunk is ``(seq_len,)``
        int64, ``cu_seqlens`` lists the per-chunk segment boundaries, ``labels``
        is the per-chunk next-token target, ``n_pad`` is always 0 for stream.
    """
    stream: list[int] = []
    for doc in documents:
        stream.extend(doc)
        if eos_id is not None:
            stream.append(eos_id)

    arr = np.asarray(stream, dtype=np.int64)
    n_chunks = len(arr) // seq_len
    if n_chunks == 0:
        raise ValueError(
            f"Not enough tokens ({len(arr)}) for a single packed sequence of "
            f"length {seq_len}. Provide more data or a shorter seq_len."
        )
    body = arr[: n_chunks * seq_len].reshape(n_chunks, seq_len)

    labels = np.empty_like(arr)
    labels[:-1] = arr[1:]
    labels[-1] = pad_id  # corpus end has no successor

    chunks: list[np.ndarray] = [body[i] for i in range(n_chunks)]
    label_chunks: list[np.ndarray] = [
        labels[i * seq_len : (i + 1) * seq_len] for i in range(n_chunks)
    ]
    cu_seqlens = [
        _cu_seqlens_stream(body[i], eos_id, seq_len) for i in range(n_chunks)
    ]
    return chunks, cu_seqlens, label_chunks, 0


def _split_long_items(items: list[np.ndarray], seq_len: int) -> list[np.ndarray]:
    """Chop items longer than ``seq_len`` into ``seq_len``-sized pieces.

    Each piece is treated as an independent segment (the same data loss model as
    stream packing, but no token is ever discarded).
    """
    out: list[np.ndarray] = []
    for item in items:
        if len(item) <= seq_len:
            out.append(item)
        else:
            for s in range(0, len(item), seq_len):
                out.append(item[s : s + seq_len])
    return out


def pack_bins(
    documents: Sequence[Sequence[int]],
    seq_len: int,
    eos_id: int | None,
    pad_id: int,
) -> tuple[list[np.ndarray], list[list[int]], list[np.ndarray], int]:
    """Pack whole documents into ``seq_len`` bins via first-fit-decreasing.

    Every document (``+<eos>``) is kept intact; documents are sorted by length
    (descending) and each is placed into the first bin with enough remaining
    room. Bins are padded with ``pad_id`` only up to ``seq_len``. Within a bin
    attention is block-diagonal over the ``doc+eos`` segments; padding is masked
    out and its labels are ``pad_id`` (ignored by the loss).

    Returns:
        ``(bins, cu_seqlens, labels, n_pad)`` — same contract as
        :func:`pack_stream`; ``n_pad`` counts the padding tokens.
    """
    items = [
        np.concatenate([np.asarray(doc, dtype=np.int64), [eos_id]])
        if eos_id is not None
        else np.asarray(doc, dtype=np.int64)
        for doc in documents
    ]
    items = _split_long_items(items, seq_len)

    bins: list[list[int]] = []
    bin_lens: list[int] = []
    for item in sorted(items, key=len, reverse=True):
        placed = False
        for i in range(len(bins)):
            if bin_lens[i] + len(item) <= seq_len:
                bins[i].extend(int(t) for t in item)
                bin_lens[i] += len(item)
                placed = True
                break
        if not placed:
            bins.append([int(t) for t in item])
            bin_lens.append(len(item))

    out_bins: list[np.ndarray] = []
    out_labels: list[np.ndarray] = []
    out_cu_seqlens: list[list[int]] = []
    n_pad = 0
    for bin_tokens, real in zip(bins, bin_lens):
        arr = np.asarray(bin_tokens, dtype=np.int64)
        padded = (
            np.pad(arr, (0, seq_len - real), constant_values=pad_id)
            if real < seq_len
            else arr
        )
        n_pad += seq_len - real
        out_bins.append(padded)
        out_labels.append(_bin_labels(arr, real, eos_id, pad_id, seq_len))
        out_cu_seqlens.append(_cu_seqlens_bin(arr, eos_id, real))
    return out_bins, out_cu_seqlens, out_labels, n_pad


def _bin_labels(
    tokens: np.ndarray, real: int, eos_id: int | None, pad_id: int, seq_len: int
) -> np.ndarray:
    """Per-segment next-token labels for a bin-packed sequence.

    Within each ``doc(+eos)`` segment, ``label[i] = token[i+1]`` — so the last
    real token of a document predicts ``<eos>``, teaching the model document
    boundaries. The ``<eos>`` token itself (and every padding slot) gets
    ``pad_id``: its successor would be a *different* document or padding, which
    the block-diagonal mask forbids attending to, so it is ignored.
    """
    labels = np.full(seq_len, pad_id, dtype=np.int64)
    seg_start = 0
    for p in range(real):
        if p == real - 1 or (eos_id is not None and int(tokens[p]) == eos_id):
            # tokens[seg_start .. p] form one segment; shift within it.
            labels[seg_start:p] = tokens[seg_start + 1 : p + 1]
            labels[p] = pad_id  # segment end (eos or last token) → ignore
            seg_start = p + 1
    return labels


def _cu_seqlens_bin(tokens: np.ndarray, eos_id: int | None, real: int) -> list[int]:
    """Segment boundaries over the *real* part of a bin-packed sequence."""
    starts = [0]
    if eos_id is not None:
        for p in range(1, real):
            if int(tokens[p - 1]) == eos_id:
                starts.append(p)
    starts.append(real)
    return starts


def pack_documents(
    documents: Sequence[Sequence[int]],
    seq_len: int,
    *,
    eos_id: int | None,
    pad_id: int,
    strategy: str = STREAM,
) -> tuple[list[np.ndarray], list[list[int]], list[np.ndarray], int]:
    """Pack documents into fixed-length sequences under the given strategy."""
    if strategy == STREAM:
        return pack_stream(documents, seq_len, eos_id, pad_id)
    if strategy == BIN:
        return pack_bins(documents, seq_len, eos_id, pad_id)
    raise ValueError(f"Unknown packing strategy {strategy!r}; use {STREAM!r} or {BIN!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Mask / position builders
# ──────────────────────────────────────────────────────────────────────────────

def build_attention_mask(
    cu_seqlens: Sequence[int], seq_len: int, n_pad: int = 0
) -> np.ndarray:
    """Block-diagonal causal attention mask for one packed sequence.

    ``mask[i, j]`` is ``True`` iff tokens ``i`` and ``j`` share a document
    segment **and** ``j <= i`` (causal). Padding positions (which lie beyond
    the final segment boundary) can attend only to themselves — this avoids
    the ``softmax([-inf, …]) = NaN`` pathology that would arise from fully
    blocking a row while still running it through the transformer. Padding
    labels are ``<pad>`` (id 0) and ignored by the loss. ``n_pad`` is
    informational; the padding follows from ``cu_seqlens`` itself. Returned
    shape ``(1, seq_len, seq_len)``.
    """
    seg = np.full(seq_len, -1, dtype=np.int64)
    for s in range(len(cu_seqlens) - 1):
        seg[cu_seqlens[s] : cu_seqlens[s + 1]] = s

    tril = np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))
    seg_col = seg[:, None]
    mask = tril & (seg_col == seg[None, :]) & (seg_col >= 0) & (seg[None, :] >= 0)
    # Allow padding positions to attend to themselves (diagonal self-loop) to
    # prevent ``softmax(all -inf) = NaN`` in the transformer while their
    # contribution is still ignored by the loss (pad label → ignore_index).
    mask |= np.eye(seq_len, dtype=np.bool_)
    return mask[None, :, :]


def build_position_ids(
    cu_seqlens: Sequence[int], seq_len: int, n_pad: int = 0
) -> np.ndarray:
    """RoPE positions for one packed sequence, reset at every segment start.

    A document's tokens occupy positions ``0..len-1`` relative to the document
    start, so rotary frequencies never jump across a document boundary.
    Padding positions are left at 0 (they are masked out of attention).
    """
    pos = np.zeros(seq_len, dtype=np.int64)
    for s in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[s], cu_seqlens[s + 1]
        pos[start:end] = np.arange(end - start)
    # padding positions (beyond the final boundary) remain 0
    return pos


# ──────────────────────────────────────────────────────────────────────────────
# Batch container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PackedBatch:
    """One packed training batch — everything the model needs for packing.

    The model consumes ``input_ids`` / ``labels`` exactly like a classic
    ``(x, y)`` pair and additionally receives ``attention_mask`` (block-diagonal
    causal) and ``position_ids`` (per-segment RoPE positions) as keyword args.
    ``cu_seqlens`` is metadata (segment boundaries per packed sequence).
    """

    input_ids: torch.Tensor       # (B, seq_len) long — packed token stream
    labels: torch.Tensor          # (B, seq_len) long — pad at ignored positions
    attention_mask: torch.Tensor  # (B, 1, seq_len, seq_len) bool — block-diagonal causal
    position_ids: torch.Tensor    # (B, seq_len) long — positions reset per segment
    cu_seqlens: tuple[tuple[int, ...], ...]  # per-sample segment boundaries
    n_pad: int = 0                # padding tokens in this batch
    real_tokens: int = 0          # non-pad token count in this batch
    n_documents: int = 0          # documents packed into this batch

    def to(self, device, non_blocking: bool = False) -> "PackedBatch":
        return PackedBatch(
            self.input_ids.to(device, non_blocking=non_blocking),
            self.labels.to(device, non_blocking=non_blocking),
            self.attention_mask.to(device, non_blocking=non_blocking),
            self.position_ids.to(device, non_blocking=non_blocking),
            self.cu_seqlens,
            self.n_pad,
            self.real_tokens,
            self.n_documents,
        )

    @property
    def model_kwargs(self) -> dict[str, torch.Tensor]:
        """Keyword args forwarded to ``MetisLM.forward``."""
        return {
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
        }

    @property
    def padding_waste_pct(self) -> float:
        """Percent of this batch's slots that are padding (0 for stream)."""
        total = self.input_ids.numel()
        return 0.0 if total == 0 else 100.0 * self.n_pad / total


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PackedDataset(IterableDataset):
    """Iterable dataset that yields dynamically packed :class:`PackedBatch`es.

    Documents are packed once at construction into fixed-length sequences; every
    epoch re-shuffles the packed-sequence order so batches are re-composed
    dynamically. Each yielded batch has exactly ``batch_size`` packed sequences
    of length ``seq_len`` (ragged remainder is dropped, like ``drop_last``).
    """

    def __init__(
        self,
        documents: Sequence[Sequence[int]],
        seq_len: int,
        batch_size: int,
        *,
        eos_id: int | None,
        pad_id: int,
        strategy: str = STREAM,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2 for packing, got {seq_len}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.strategy = strategy
        self.shuffle = shuffle
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self.packed, self.cu_seqlens, self.labels, self.n_pad = pack_documents(
            documents, seq_len, eos_id=eos_id, pad_id=pad_id, strategy=strategy,
        )
        self.n_sequences = len(self.packed)
        self.n_batches = self.n_sequences // batch_size
        if self.n_batches == 0:
            raise ValueError(
                f"Only {self.n_sequences} packed sequence(s) — fewer than "
                f"batch_size={batch_size}. Provide more data or shrink the batch."
            )

        # Metadata for statistics / benchmarks.
        total_slots = self.n_sequences * seq_len
        self.n_documents = len(documents)
        self.total_real_tokens = total_slots - self.n_pad
        self.padding_waste_pct = 100.0 * self.n_pad / total_slots

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        order = np.arange(self.n_sequences)
        if self.shuffle:
            self._rng.shuffle(order)
        for b in range(self.n_batches):
            idx = order[b * self.batch_size : (b + 1) * self.batch_size]
            yield self._build_batch(idx)

    def _build_batch(self, idx: np.ndarray) -> PackedBatch:
        B = len(idx)
        T = self.seq_len
        input_ids = np.stack([self.packed[i] for i in idx], axis=0)
        labels = np.stack([self.labels[i] for i in idx], axis=0)

        masks = np.stack(
            [build_attention_mask(self.cu_seqlens[i], T) for i in idx], axis=0
        )  # (B, 1, T, T)
        positions = np.stack(
            [build_position_ids(self.cu_seqlens[i], T) for i in idx], axis=0
        )  # (B, T)

        cu_seqlens = tuple(tuple(cs) for cs in (self.cu_seqlens[i] for i in idx))
        n_pad_batch = int((input_ids == self.pad_id).sum())
        real = B * T - n_pad_batch
        return PackedBatch(
            input_ids=torch.from_numpy(input_ids).long(),
            labels=torch.from_numpy(labels).long(),
            attention_mask=torch.from_numpy(masks).bool(),
            position_ids=torch.from_numpy(positions).long(),
            cu_seqlens=cu_seqlens,
            n_pad=n_pad_batch,
            real_tokens=real,
            n_documents=self.n_documents,
        )

    def statistics(self) -> dict[str, float]:
        """Pack efficiency statistics for benchmarking / reporting."""
        docs_per_seq = self.n_documents / max(1, self.n_sequences)
        return {
            "strategy": self.strategy,
            "n_documents": float(self.n_documents),
            "n_packed_sequences": float(self.n_sequences),
            "n_batches": float(self.n_batches),
            "seq_len": float(self.seq_len),
            "total_slots": float(self.n_sequences * self.seq_len),
            "total_real_tokens": float(self.total_real_tokens),
            "padding_waste_pct": self.padding_waste_pct,
            "avg_documents_per_sequence": docs_per_seq,
        }
