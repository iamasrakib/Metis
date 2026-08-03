# Dynamic Sequence Packing

Training sequences (documents) in a language-model corpus vary in length. The
traditional fixed-length pipeline wastes the tails of every batch: short
sequences are right-padded with `<pad>` tokens that contribute no signal while
still paying for full attention. Dynamic sequence packing eliminates that waste
by combining several short sequences into every packed batch.

## Two strategies

### `"stream"` — contiguous 1D packing (default)

Documents are concatenated with `<eos>` separators into one contiguous token
stream, then chopped into fixed-length sequences. A sequence boundary may fall
mid-document; every token's next-token target stays valid because the corpus is
contiguous, so there is **zero padding waste**.

Within each packed sequence, attention is block-diagonal: a token may only
attend to earlier tokens of the same document (segment). The `<eos>` token
marks segment boundaries.

### `"bin"` — whole-document first-fit-decreasing

Documents (with `<eos>` appended) are sorted by length descending, then each is
placed into the first bin with enough room. No document is ever split. Bins are
padded with `<pad>` only up to `max_seq_len`, minimising waste while keeping
every document fully visible.

Both strategies produce fixed `(B, T)` tensors, so they work with the existing
model shape constraints and require no special kernel support.

## What the model sees

A packed batch is a [`PackedBatch`](../metis/packing.py) carrying:

| Field | Shape | Purpose |
|-------|-------|---------|
| `input_ids` | `(B, T)` | Packed token stream (padding at tails in bin mode) |
| `labels` | `(B, T)` | Next-token targets; padding positions are `<pad>` (id 0), ignored by the loss |
| `attention_mask` | `(B, 1, T, T)` | Block-diagonal causal mask: `True` iff same segment and `j ≤ i` |
| `position_ids` | `(B, T)` | RoPE positions reset to 0 at every segment start |
| `cu_seqlens` | list of `(K+1)` tuples | Cumulative segment boundaries per packed sequence (metadata / varlen kernels) |

The model's `forward()` accepts `attention_mask` and `position_ids` as
optional keyword arguments. When they are absent (the default), behaviour is
identical to the standard fixed-length pipeline — so every existing checkpoint
and generation path is unchanged.

### Attention mask

The mask is a bool tensor `True` where attention is allowed. For the
`"stream"` strategy every position belongs to exactly one segment and padding
waste is 0, so the mask is purely block-diagonal. For `"bin"`, padding rows
get a diagonal self-loop (to prevent `softmax(all -inf) = NaN`) and their
labels are `<pad>` (ignored by the loss). The mask is built on the host per
batch and moved to GPU; for `T = 512`, `B = 8` that is ~2 MB per batch.

### RoPE position reset

Each document segment restarts its rotary position clock at 0 so that
high-frequency components never jump across a document boundary. The mask
already prevents cross-segment attention; the position reset ensures the
positional encoding is consistent within each segment.

### Loss and labels

The loss is `F.cross_entropy(..., ignore_index=0)` — identical to the existing
pipeline. Padding tokens and segment-final tokens (after `<eos>`) are
labelled `<pad>` (id 0) and ignored. The model only trains on real tokens.

## CLI usage

```bash
# Enable stream packing (default)
metis train --dataset data/input.txt --use-packing

# Use bin packing
metis train --dataset data/input.txt --use-packing --packing-strategy bin

# Combine with other flags
metis train --preset small --use-packing --use-ema --tokenizer cl100k_base
```

## Python API

```python
from metis.data import create_packed_dataloader, load_documents, split_documents
from metis.data import BPETokenizer

docs = load_documents("data/input.txt")
train_docs, val_docs = split_documents(docs, 0.9)
tokenizer = BPETokenizer("cl100k_base")
tokenizer.fit("\n\n".join(train_docs))

loader = create_packed_dataloader(
    train_docs, tokenizer,
    seq_len=512, batch_size=8,
    strategy="stream",  # or "bin"
    shuffle=True, seed=42,
)
for batch in loader:
    batch = batch.to("cuda")
    _, loss, _ = model(
        batch.input_ids, batch.labels,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
    )
```

## Correctness

Three key properties are verified in `tests/test_packing.py`:

1. **`eos=None` stream packing is bit-identical to the standard causal
   path** — when documents are concatenated without separators and attention
   covers all tokens, the mask and position tensors reproduce the plain
   causal computation exactly.

2. **Bin-packed loss equals the token-weighted per-segment standalone loss** —
   the model output for each packed segment is identical to running the
   segment in isolation, confirming that RoPE positions, attention masking,
   and loss averaging are all correct.

3. **Mask exactness** — for every position pair `(i, j)` the mask is `True`
   iff `seg(i) = seg(j)` and `j ≤ i`.

## Limitations and notes

- **Cross-boundary stream packing**: when a sequence boundary falls
  mid-document, the model cannot attend to the beginning of that document
  (the context is in the previous packed sequence). This is the standard
  1D packing trade-off; it is statistically negligible for long corpora.

- **CUDA Graphs are disabled** when packing is active, because the
  packed masks are data-shaped tensors that change per batch. Training
  runs the eager (non-graphed) micro-batch loop. This is documented and
  the fallback is logged.

- **Attention Sink** (`use_attention_sink`) is incompatible with packing.
  The two flags cannot be enabled simultaneously; the config raises
  `ValueError` at construction.

- **Tokenizer interface** is fully preserved: documents are ordinary
  token-ID lists; `<eos>` separators and `<pad>` tokens come from the
  tokenizer's `eos_id` / `pad_id` properties; the loss `ignore_index` is
  pad (id 0), matching the existing pipeline.

## Benchmarks

Run the packing benchmark:

```bash
python benchmarks/benchmark_packing.py                    # auto device, all modes
python benchmarks/benchmark_packing.py --mode throughput  # throughput only
python benchmarks/benchmark_packing.py --mode memory      # GPU peak memory
```

Results are written as JSON + Markdown under `benchmarks/results/`.

Typical results (RTX 2050, 4.3 GB VRAM, small preset, 200 docs):

| Strategy | Waste % | Tokens/s | Notes |
|----------|--------:|----------:|-------|
| packed (stream) | 0.0% | highest | zero padding, block-diagonal mask |
| packed (bin) | ~5-15% | high | minimal padding, full doc visibility |
| padded baseline | ~30-50% | lowest | wasted compute on pad tokens |
