<div align="center">

# Μῆτις (Metis) v3.0

### A Modern Tiny Language Model — Built From Scratch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)
[![CI](https://github.com/iamasrakib/metis/actions/workflows/ci.yml/badge.svg)](https://github.com/iamasrakib/metis/actions/workflows/ci.yml)

*Named after the Greek Titaness of wisdom and deep thought (Μῆτις)*

**Train a real language model on your own data in minutes — on a single GPU or even just a CPU.**

---

[Features](#features) · [Quick Start](#quick-start) · [Architecture](#architecture) · [Python API](#python-api) · [Training](#training) · [Generation](#generation) · [Serving](#serving) · [Web UI](#web-ui) · [Configuration](#configuration) · [Notebook](#notebook)

</div>

---

## Features

| Category | Details |
|----------|---------|
| **Architecture** | Decoder-only Transformer with RMSNorm, RoPE, SwiGLU, **GQA**, optional **MoE**, optional **QK-Norm**, optional **Attention Sink** |
| **Training** | AMP, gradient accumulation, gradient clipping, cosine LR with warmup, **Distributed Data Parallel (DDP)**, **EMA**, **LR Range Finder**, **W&B tracking** |
| **Inference** | KV-cache, temperature/top-k/top-p sampling, repetition penalty, streaming, **REST API server**, **Gradio Web UI**, **OpenAI-compatible endpoints** |
| **KV Cache** | Optional backends — **static** preallocated buffers (bit-identical), **quantized** int8 compressed cache (~3.8x memory cut, near-lossless), **MLA** Multi-head Latent Attention (see `docs/kv_cache.md` / `docs/mla.md`) |
| **Tokenizer** | **BPE (tiktoken)** with character-level fallback, pre-tokenization **caching**, multi-file directory loading |
| **Data** | **Memory-mapped datasets** for GB-scale corpora, streaming iterable dataset, tokenization cache, **dynamic sequence packing** (zero-padding 1D / whole-doc bin) |
| **Memory** | Gradient checkpointing, weight tying, GQA, optional **Flash Attention v2**, **persistent expert cache** (MoE weights stay resident) |
| **Scheduling** | **Graph-based execution scheduler** — computation-graph analysis, operator cost model, safe reorder, liveness-based buffer reuse, zero-sync infer path (see `docs/exec_scheduler.md`) |
| **Usability** | Unified `metis` CLI with 8 commands, interactive chat, structured logging, **test suite** |
| **Package** | Clean `from metis import ...` API, pip-installable, CI pipeline

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
pip install -e .                    # install the metis CLI
```

### 2. Prepare Data

Place a `.txt` file in the `data/` directory, or use the multi-file directory loading:

```bash
# Example: download Tiny Shakespeare
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'data/input.txt')"

# Or: generate the Siraj-ud-Daulah knowledge dataset (see below)
python data/generate_siraj_dataset.py
```

**Multi-file datasets:** Pass a directory — all `.txt` files are concatenated:
```bash
metis train --dataset data/   # loads data/*.txt
```

### 3. Train

```bash
# Train with defaults (small preset, char tokenizer)
metis train --dataset data/input.txt

# Use BPE tokenizer (cl100k_base = GPT-4 tokenizer)
metis train --tokenizer cl100k_base --dataset data/input.txt

# Use a preset, enable MoE and QK-Normalization
metis train --preset medium --iters 10000 --use-moe --use-qk-norm

# Enable EMA and W&B tracking
metis train --preset small --use-ema --use-wandb

# Enable dynamic sequence packing (zero padding waste)
metis train --preset small --use-packing --tokenizer cl100k_base

# Resume from checkpoint
metis train --resume

# Learning rate range finder
metis find-lr --preset tiny --dataset data/input.txt
```

### 4. Chat

```bash
metis chat           # interactive terminal chat
metis ui             # browser-based Gradio chat interface
```

### 5. Generate

```bash
metis generate --prompt "Once upon a time" --max-tokens 300 --temperature 0.7
```

### 6. REST API Server

```bash
metis serve          # starts FastAPI server on port 8000
# Docs: http://localhost:8000/docs
# Chat: POST http://localhost:8000/chat
# OpenAI-compatible: POST http://localhost:8000/v1/chat/completions
```

### 7. Check model status

```bash
metis info           # show checkpoint directory & config
```

---

## Architecture

Metis implements a modern decoder-only transformer following best practices from the LLaMA / Mistral / GPT family, with optional MoE, QK-Norm, and Attention Sink extensions.

```
Input Tokens
     │
     ▼
┌─────────────┐
│  Token Emb  │   (no position embedding when using RoPE)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         Transformer Block  ×N           │
│  ┌───────────────────────────────────┐  │
│  │  RMSNorm → Multi-Head Attention   │  │
│  │  (with RoPE & Causal Mask)        │  │
│  │  + Residual Connection            │  │
│  ├───────────────────────────────────┤  │
│  │  RMSNorm → SwiGLU FFN            │  │
│  │  + Residual Connection            │  │
│  └───────────────────────────────────┘  │
└────────────────┬────────────────────────┘
                 │
                 ▼
          ┌─────────────┐
          │   RMSNorm    │
          └──────┬──────┘
                 │
                 ▼
          ┌─────────────┐
          │   LM Head    │  (weight-tied with token embedding)
          └──────┬──────┘
                 │
                 ▼
         Output Logits
```

### Key Components

| Component | Description | Reference |
|-----------|-------------|-----------|
| **RMSNorm** | Root Mean Square normalization — simpler and faster than LayerNorm | [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467) |
| **RoPE** | Rotary Position Embeddings — encode position as rotation in embedding space | [Su et al., 2021](https://arxiv.org/abs/2104.09864) |
| **SwiGLU** | Gated feed-forward with SiLU activation — more expressive per parameter | [Shazeer, 2020](https://arxiv.org/abs/2002.05202) |
| **GQA** | Grouped Query Attention — fewer KV heads than query heads for efficient inference | [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245) |
| **QK-Norm** | Query/Key normalization for stable training at high learning rates | [Dehghani et al., 2023](https://arxiv.org/abs/2305.17303) |
| **MoE** | Mixture of Experts — sparse FFN with grouped, dynamically-scheduled execution (token sorting → expert grouping → dynamic capacity → grouped GEMM), up to 2.4× faster than the legacy scheduler under skewed routing. **Persistent expert cache** keeps active expert weights resident (90% hit rate, 90% bandwidth reduction on inference), and **layer prefetching** speculatively warms the *next* layer's cache on a side stream during compute — see [docs/layer_prefetch.md](docs/layer_prefetch.md) | [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538) |
| **KV-Cache** | Cache key/value states during generation for O(1) per-token inference | Standard practice |
| **BPE** | Byte-Pair Encoding tokenizer via tiktoken — 4× better compression than character-level | [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909) |

---

## FlashAttention-2

Metis runs all causal attention (training, validation, and KV-cache inference)
through a FlashAttention-2 dispatch layer that picks the fastest kernel
available on the machine and **falls back automatically** when none is:
fused kernels on CUDA → PyTorch `scaled_dot_product_attention` (FA2 / 
memory-efficient) → exact manual math. No code changes at the call site, no
extra dependencies required, and an **unchanged public API**.

| Backend | How to select | Fused |
|---------|---------------|-------|
| dao-AILab `flash-attn` package (Linux) | `flash_attn` | ✅ |
| torch SDPA FA2 kernel | `flash` | ✅ |
| torch SDPA memory-efficient kernel | `mem_efficient` | ✅ |
| best available (default) | `auto` | ✅ |
| exact manual math (deterministic) | `math` | ❌ |

Select via `ModelConfig(attn_backend=...)`, the `METIS_ATTN_BACKEND` env var,
or `--attn-backend` on `metis train/generate/chat/info/serve`. `use_flash_attn=False`
pins the exact manual reference.

```python
from metis import detect_attention_backends, fused_attention_supported

detect_attention_backends()   # GPU capability + per-kernel availability + "recommended"
fused_attention_supported()   # is a fused kernel usable on this machine?
```

At long sequences the fused kernels cut activation memory from O(T²) to O(T) —
on the RTX 2050 reference run, a T=2048 training step uses **736 MB (math) vs
171 MB (fused, −77%)**, and prefill attention runs up to **8.5× faster**.
Run the benchmark yourself:

```bash
python benchmarks/benchmark_attention.py               # kernel + model + memory
python benchmarks/benchmark_attention.py --mode memory  # peak activation memory
```

Full design, numerical guarantees, feature matrix, and results:
**[docs/flash_attention.md](docs/flash_attention.md)**.

---

## Fused Block

The Transformer block minimises kernel launches by structurally fusing hot
paths in eager PyTorch:

| Fusion | What changed | Kernel reduction |
|--------|-------------|-----------------|
| **RMSNorm** | `F.rms_norm` (single fused kernel) replaces the manual 7-op chain | ~6 ops → 1 |
| **QKV projection** | One `nn.Linear(d, d + 2·kv)` replaces three separate GEMMs | 3 → 1 GEMM |
| **RoPE** | `apply_rope_pair(q, k, …)` rotates q and k jointly | 6 → 4 kernels |
| **SwiGLU gate/up** | One `nn.Linear(d, 2h)` replaces `w1` + `w3` | 2 → 1 GEMM |

**Result on RTX 2050 (bf16 autocast, d=256, T=128):**

- block forward+backward: **4.02 ms** (21% faster vs pre-fusion baseline)
- decode (single-token, 128-token cache): **1.32 ms**, 1.1 GB/s effective BW

All fusions are **bit-identical** to their pre-fusion forms — verified by
`benchmarks/verify_block_parity.py` across fp32, bf16, and fp16, including
gradients, KV-cache decode, gradient checkpointing, and a real pre-fusion
checkpoint load. Existing checkpoints load byte-identically via a state-dict
compatibility shim (old `q_proj/k_proj/v_proj` and `w1/w3` keys preserved).

The output-projection + residual path stays as separate kernels; fusing them
into the attention call requires Triton (with it, `torch.compile` does this
automatically).

Full design, numerical guarantees, and measured results:
**[docs/fused_block.md](docs/fused_block.md)**.

---

## Persistent Expert Cache

MoE models cache their stacked + dtype-cast expert weight tensors so active
experts stay **resident in GPU memory** instead of being re-stacked from the
fp32 master weights on every forward. On by default (`moe_cache_size=64`),
active in both training and inference, auto-invalidated after every optimizer
step, and resilient to stale mutations via staleness signatures (data_ptr +
_version + shape).

Under AMP autocast, the cache pre-casts to the compute dtype (bf16/fp16)
rather than fp32, halving resident memory and eliminating the per-bmm
autocast weight cast.

| Metric | Value (RTX 2050, 4 layers, 8 experts) |
|--------|----------------------------------------|
| Cache hit rate (same-input eval forwards) | **95.0%** |
| Stack+cast rematerialization avoided | **95.0%** |
| Total MoE weight-traffic reduction | **84.0%** |
| Resident expert memory | **11.3 MB** |
| Numerical parity | bit-identical (`torch.equal`) to uncached |

Outputs are bit-identical to the uncached path, so existing checkpoints and
workflows are unaffected. Tune it via config (`moe_cache_size`,
`moe_cache_bytes`) or CLI flags (`--moe-cache-size`, `--moe-cache-bytes`).

Full design, staleness semantics, and measured results:
**[docs/expert_cache.md](docs/expert_cache.md)**.

---

## KV Cache Subsystem

The KV cache is pluggable via the `kv_backend` flag (config or CLI
`--kv-backend` / `METIS_KV_BACKEND`). All backends preserve the public API —
`model.forward(idx, ..., kv_cache=...) -> (logits, loss, new_kv_cache)` — and
the cache object round-trips opaquely through `generate_text`, the server and
the scheduler.

| Backend | Description | Output quality | Cache memory (per layer, T=512, small preset) |
|---------|-------------|----------------|-----------------------------------------------|
| `default` | legacy growable `(K, V)` tuples via `torch.cat` per step | bit-identical reference | 512 KB |
| `static` | preallocated contiguous buffers, in-place writes, flat memory | **bit-identical** (`torch.equal`) | 512 KB (flat, no per-step alloc) |
| `quantized` | static layout + int8 K/V, per-token scales | near-lossless (max logit diff `6e-3`) | **136 KB (3.8x less)** |
| `mla` | Multi-head Latent Attention (architecture change — train from scratch) | n/a (different weights) | 384 KB (1.3x, grows with `n_heads`) |

The **quantized** backend delivers the headline result: ~4x cache-memory
reduction with a max logit deviation of `6e-3` — far below the sampling noise
of a temperature > 0 generation. The **static** backend is a pure layout
optimisation with zero numerical change. **MLA** is a research-grade
architecture swap (DeepSeek-V2/V3 style) that compresses the KV state into a
learned latent, verified algebraically identical to explicit attention.

Quick usage:

```bash
metis generate --kv-backend static     # preallocated buffers (bit-identical)
metis generate --kv-backend quantized  # int8 compressed cache (~4x memory cut)
metis generate --kv-backend mla        # latent attention (train from scratch first)
```

Or in code:

```python
from metis import ModelConfig, MetisLM, generate_text
config = ModelConfig.from_preset("small", kv_backend="quantized")
model = MetisLM(config)          # train with this config
out = generate_text(model, tok, "Once upon a time")
```

Verification and benchmarks:
- `benchmarks/verify_kv_parity.py` — static is `torch.equal`-identical to
  default; quantized error bounded; MLA absorbed-vs-explicit `5.8e-6`.
- `benchmarks/benchmark_kv.py` — memory + throughput comparison across all
  backends, writes `benchmarks/results/`.
- `tests/test_kv.py` — 44 unit tests (quantization round-trip, LayerKV/KVCache
  lifecycle, cached_len_of, memory formulas, config validation, model parity).

Full design and measured results:
**[docs/kv_cache.md](docs/kv_cache.md)** · **[docs/mla.md](docs/mla.md)**.

---

## Python API

```python
from metis import MetisLM, ModelConfig, CharTokenizer, generate_text, PRESETS

# Build config from a preset
config = ModelConfig.from_preset("medium", max_iters=10000)
print(config.summary())

# Create tokenizer and model
tokenizer = CharTokenizer()
tokenizer.fit("Some text...")
model = MetisLM(config)

# Generate text
output = generate_text(
    model, tokenizer, "Once upon a time",
    max_new_tokens=200, temperature=0.8,
    top_k=40, top_p=0.9,
)

# Full training pipeline
from metis import train
train(config, resume=False)
```

---

## Training

### Model Presets

| Preset | d_model | Heads | Layers | Seq Len | ~Params | VRAM  |
|--------|---------|-------|--------|---------|---------|-------|
| `tiny` | 128 | 4 | 4 | 256 | ~1M | <1 GB |
| `small` | 256 | 4 | 4 | 256 | ~4M | ~1 GB |
| `medium` | 384 | 6 | 6 | 512 | ~15M | ~3 GB |
| `large` | 512 | 8 | 8 | 512 | ~35M | ~6 GB |
| `0.5b` | 1536 | 16 | 12 | 1024 | ~0.47B | ~4 GB (bnb8bit) — comfortable on a 16 GB T4 |
| `1b` | 2048 | 16 | 16 | 1024 | ~1.01B | 16 GB (tight — requires bnb8bit) |

### CLI Commands

| Command | Description |
|---------|-------------|
| `metis train` | Train a model (with BPE/char tokenizer, MoE, DDP, EMA, W&B) |
| `metis distill` | Distill from an API teacher — train forever, auto-resume |
| `metis generate` | Generate text from a single prompt |
| `metis chat` | Interactive terminal chat |
| `metis serve` | REST API server (FastAPI, OpenAI-compatible) |
| `metis ui` | Gradio web UI in browser |
| `metis info` | Model & checkpoint status |
| `metis find-lr` | Learning rate range finder |

```
metis train [OPTIONS]

Advanced options:
  --dataset PATH        Path to training text file (default: data/input.txt)
  --preset NAME         Model preset: tiny / small / medium / large / 0.5b / 1b
  --iters N             Max training iterations (default: 5000)
  --lr FLOAT            Peak learning rate (default: 3e-4)
  --batch-size N        Micro batch size (default: 8)
  --grad-accum N        Gradient accumulation steps (effective batch = batch-size × grad-accum)
  --seq-len N           Max sequence length (default: 256)
  --optimizer NAME      Optimizer: adamw (default) / bnb8bit (bitsandbytes 8-bit Adam —
                        fits ~1B-param models in 16 GB VRAM; falls back to AdamW if
                        bitsandbytes isn't installed)
  --tokenizer NAME      Tokenizer: char / cl100k_base / p50k_base / o200k_base
  --resume              Resume from latest checkpoint
  --compile             Use torch.compile (PyTorch 2.0+)
  --attn-backend NAME   Attention backend: auto / flash_attn / sdpa / flash / mem_efficient / math
  --use-moe             Enable Mixture of Experts
  --moe-engine NAME     MoE engine: auto / grouped (default) / per_expert
  --moe-group-ratio N   Expert-group max/min token ratio (default 2.0; lower =
                        tighter blocks, more bmm pairs; 1e9 = single group)
  --moe-cache-size N    Max entries in the persistent expert weight cache
                        (default 64; 0 = disabled)
  --moe-cache-bytes N   Optional byte budget for the expert cache
                        (default 0 = unbounded)
  --use-qk-norm         Enable QK-Normalization
  --use-ema             Enable Exponential Moving Average
  --use-wandb           Log metrics to Weights & Biases
  --use-packing         Enable dynamic sequence packing (zero padding waste)
  --packing-strategy    stream (default) or bin (whole-doc FFD)
  --num-workers N       DataLoader workers (default: 0)
  --seed N              Random seed (default: 42)
  --log-level LEVEL     DEBUG / INFO / WARNING / ERROR
```

### Training Features

- **Cosine LR decay** with linear warmup
- **Gradient accumulation** (effective batch = micro_batch × accum_steps)
- **Gradient clipping** (default: max_norm=1.0)
- **Automatic Mixed Precision** on CUDA (FP16 / BF16)
- **Distributed Data Parallel (DDP)** — multi-GPU training
- **Exponential Moving Average (EMA)** of model weights
- **Weights & Biases** experiment tracking
- **Learning Rate Range Finder** (`metis find-lr`)
- **Periodic validation** with best-model tracking
- **Perplexity metric** reported alongside loss at every validation
- **Sample generation** during training to monitor quality
- **Checkpointing** with full optimizer state for seamless resume
- **Memory-mapped datasets** for GB-scale corpora
- **Pre-tokenization caching** for faster data loading
- **Dynamic sequence packing** — packs short documents into dense batches (zero padding waste via 1D stream or whole-document bin packing); block-diagonal attention mask + per-segment RoPE positions
- **Overlapped training pipeline** — software-pipelines disk I/O, tokenization, CPU preprocessing, H2D copies, GPU compute, and checkpoint writes: background-thread prefetching (with pre-pinning), non-blocking H2D on a dedicated copy stream with cross-step overlap, an async checkpoint snapshot whose D2H overlaps the next step (`submit_async` + `wait_pending`, tear-free), and a `GpuIdleTracker` reporting GPU idle % **and per-stage wall attribution** (`data_wait` / `h2d` / `compute` / `optimizer` / `checkpoint`) per step (on by default; `--no-pipeline` to disable) — see [docs/pipeline.md](docs/pipeline.md)

---

## Siraj-ud-Daulah Knowledge Dataset

Train a model on the history of Nawab Siraj-ud-Daulah, the last independent Nawab of Bengal (1756–1757). The dataset covers:

- **Biography** — early life, ascension to the throne, family background
- **Conflict with the British** — capture of Calcutta, dastak trade abuses
- **Battle of Plassey** (1757) — the conspiracy, betrayal by Mir Jafar, Robert Clive
- **Legacy** — his place in Bengali history, the term "mirjafar"

Generate and train:

```bash
# 1. Create the dataset
python data/generate_siraj_dataset.py

# 2. Train with GQA for efficient inference (2 KV heads, 4 query heads)
metis train --dataset data/siraj_all.txt --preset tiny --n-kv-heads 2 --iters 5000

# 3. Chat about Siraj-ud-Daulah
metis chat --checkpoint-dir checkpoints
```

The dataset generator creates three files:
| File | Content | Size |
|------|---------|------|
| `data/siraj_narrative.txt` | Rich historical narrative (~8K chars) | |
| `data/siraj_qa.txt` | 27 Q&A topics × 5 repetitions (~35K chars) | |
| `data/siraj_all.txt` | Combined corpus (~44K chars) | |

---

## Generation

### Interactive Chat

```bash
metis chat
```

Chat commands:
| Command | Action |
|---------|--------|
| `/quit` or `/exit` | End conversation |
| `/clear` | Clear history |
| `/temp 0.5` | Set temperature |
| `/help` | Show help |

### Single Prompt

```bash
metis generate --prompt "Once upon a time" --max-tokens 300 --temperature 0.7
```

### CLI Options

```
metis generate [OPTIONS]

Options:
  --prompt TEXT              Generate from this prompt (required)
  --max-tokens N             Max tokens to generate (default: 200)
  --temperature FLOAT        Sampling temperature (default: 0.8)
  --top-k N                  Top-k sampling (default: 40)
  --top-p FLOAT              Nucleus sampling (default: 0.9)
  --repetition-penalty FLOAT Repetition penalty (default: 1.1)
  --no-stream                Disable streaming output
  --no-cache                 Disable KV-cache (debugging)
  --checkpoint-dir PATH      Checkpoint directory (default: checkpoints)
  --device DEVICE            Force cpu or cuda
  --attn-backend NAME        Attention backend: auto / flash_attn / sdpa / flash / mem_efficient / math
```

---

## Serving

Metis includes a FastAPI-based REST API server with OpenAI-compatible endpoints.

### Start the Server

```bash
metis serve --port 8000
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API overview & available endpoints |
| `GET /health` | Health check |
| `GET /info` | Model metadata |
| `POST /generate` | Generate text from a prompt |
| `POST /chat` | Multi-turn chat with message history |
| `POST /v1/completions` | OpenAI-compatible completions |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions |

### Web UI

```bash
metis ui --port 7860
```

Opens a Gradio-based chat interface in your browser with adjustable sampling parameters.

---

## Configuration

All settings are managed through `config.py` with full validation:

```python
from metis import ModelConfig

# From preset
config = ModelConfig.from_preset("medium", max_iters=10000)

# From JSON
config = ModelConfig.from_json("checkpoints/config.json")

# Manual
config = ModelConfig(d_model=384, n_heads=6, n_layers=6)

# Save / display
config.save_json("my_config.json")
print(config.summary())
```

---

## Training on Google Colab (free GPU)

No GPU at home? The included **`Metis_Colab_Training.ipynb`** trains Metis on Colab's
free T4 GPU and saves checkpoints straight to your **Google Drive** (a Colab disconnect
loses nothing). It downloads a **real high-quality corpus — FineWeb-Edu** (HuggingFace's
educational web corpus, ~500M tokens) and trains the **`0.5b` preset** with **8-bit Adam**
(`--optimizer bnb8bit`). At ~0.47B params it fits the T4's 16 GB with ~10 GB of headroom —
even if bitsandbytes is unavailable and training falls back to plain AdamW, it still fits.

**Run top to bottom:**
1. Mount Google Drive.
2. Clone this repo (it's public — no login needed) + install dependencies.
3. Download FineWeb-Edu (~2 GB text → `MyDrive/Metis/fineweb_edu.txt`). Runs once.
4. Pick your dataset (defaults to FineWeb-Edu; drop your own `.txt` corpus in Drive to
   override).
5. Train (`metis train --preset 0.5b --optimizer bnb8bit --no-cuda-graphs`, 4000 steps ≈
   525M tokens). Checkpoints and the tokenization cache land in
   `MyDrive/Metis/checkpoints_05b` / `cache`, so **re-running the cell resumes** exactly
   where you stopped — across days and sessions.

**Open the notebook — one click:**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iamasrakib/Metis/blob/main/Metis_Colab_Training.ipynb)

Or: Colab → **File → Open notebook → GitHub** → select `iamasrakib/Metis` →
`Metis_Colab_Training.ipynb`.

**Expectations:** the 0.5B model runs ~3–6k tokens/s on a T4 (≈2× a 1B model), so 500M
tokens takes several Colab sessions (re-run the train cell to continue). It will be
**data-limited** next to production models (which train on 200B+ tokens), but it's a real,
coherent model and the run resumes indefinitely. Raise `--iters` to train longer. OOM is
unlikely on VRAM — the run leaves ~10 GB of the T4's 16 GB free; you can even raise `--batch-size`
to 4 for more throughput.

**⚠️ RAM OOM (Colab-specific):** Colab's free tier has ~12 GB system RAM. The Phase 6
async checkpointing (on by default) snapshots the entire model+optimizer+EMA state to
CPU (~4 GB for 0.5B) and holds it there while the background thread writes to Google
Drive's slow FUSE mount — this can push RAM over the limit and kill the kernel with **no
traceback**. The notebook uses `--no-async-checkpoint` (synchronous checkpoint writes)
which frees the CPU snapshot immediately. If you ever hit a silent Colab kernel death
during training, check the last `memwatch:` line in the logs — it shows GPU alloc/reserved
+ system RAM, so you can see what the headroom was.

---

## Distillation (train forever from an API)

`metis distill` trains Metis **continuously** on text written by a frontier
teacher model (ChatGPT / DeepSeek / Claude class) reached through an
OpenAI-compatible API — the classic way to *distill* knowledge into a small
model. The loop never stops on its own; it keeps writing and learning until you
stop it. **Stopping and restarting resumes automatically — no setup.**

```bash
# 1. Point at your teacher (env vars, or --teacher-* flags)
export METIS_TEACHER_BASE_URL="https://your-gateway.example/v1"
export METIS_TEACHER_API_KEY="sk-..."
export METIS_TEACHER_MODEL="deepseek-chat"

# 2. Verify the connection with one call (adjust the contract if needed)
metis distill --test-teacher

# 3. Train forever
metis distill --checkpoint-dir checkpoints_distill --preset tiny --tokenizer cl100k_base
```

> **No public teacher?** Metis also works against a *local* OpenAI-compatible
> gateway. Expose one to Colab/remote runs with `start_tunnel.bat` /
> `stop_tunnel.bat` (a Cloudflare tunnel), then point `METIS_TEACHER_BASE_URL`
> at the tunnel URL. Teacher calls time out after `METIS_TEACHER_TIMEOUT`
> seconds (default 240).

**Stop / resume:**

- **Stop:** press `Ctrl+C` (saves first), or create a file named `STOP` in the
  checkpoint dir.
- **Resume:** re-run the same command. It reloads `latest_checkpoint.pt`,
  `distill_state.json`, and `tokenizer.json` and continues exactly where it
  stopped. This works across days and machines (and Colab reconnects if the
  checkpoint dir lives on Drive).
- A trained checkpoint dir is a normal Metis checkpoint — chat with it via
  `metis chat --checkpoint-dir checkpoints_distill`.

**How it works:** each iteration asks the teacher to write a chunk of prose
about a topic (rotating through `--topic-file` if given), then Metis trains a
few optimizer steps on that text (its normal next-token objective — true
logit-level distillation is impossible across different tokenizers, so this is
the standard "text distillation"). Tokenizer is fit **once** and reused, so the
vocabulary never drifts on an endless stream. On CUDA, mixed precision follows
`get_amp_dtype` (fp16 on Turing/T4, bf16 on Ampere+).

**Pacing / cost guards** (an infinite loop hits a paid API):

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-tokens` | 1024 | Max tokens per teacher call |
| `--min-sleep` | 1.0 | Min seconds between calls |
| `--steps-per-call` | 4 | Optimizer steps per call |
| `--budget-tokens` | 0 | Stop after this many teacher tokens (0 = unlimited) |
| `--save-every` | 50 | Checkpoint + state save interval (steps) |

Other useful flags: `--topic "animals"`, `--topic-file topics.txt`,
`--tokenizer char --seed-data corpus.txt` (fit a char vocab once), `--no-resume`
(start fresh), `--mock` (offline teacher for testing).

---

## Project Structure

```
Μῆτις/
├── metis/                    # The importable Python package
│   ├── __init__.py           # Public API (MetisLM, ModelConfig, BPETokenizer, …)
│   ├── cli.py                # Unified CLI (train / distill / generate / chat / serve / ui / info / find-lr)
│   ├── config.py             # Configuration system with presets & validation
│   ├── model.py              # Transformer core (RMSNorm, RoPE, SwiGLU, GQA, MoE, QK-Norm)
│   ├── attn.py               # FlashAttention-2 dispatch layer (auto-fallback)
│   ├── data.py               # BPETokenizer, MMapDataset, streaming, cache
│   ├── training.py           # Training loop + DDP, EMA, W&B, LR finder
│   ├── pipeline.py           # Overlapped training pipeline (prefetch / staging)
│   ├── cuda_graphs.py        # CUDA-graph capture & replay
│   ├── moe.py                # Mixture of Experts (grouped-GEMM engine)
│   ├── expert_cache.py       # Persistent expert-weight cache
│   ├── layer_prefetch.py     # Layer-expert prefetching
│   ├── packing.py            # Dynamic sequence packing
│   ├── kv.py                 # KV cache backends (default / static / quantized / MLA)
│   ├── mla.py                # Multi-head Latent Attention
│   ├── generate.py           # Generation & interactive chat
│   ├── server.py             # FastAPI REST server (OpenAI-compatible)
│   ├── webui.py              # Gradio browser chat interface
│   ├── distill.py            # Continuous distillation from an API teacher
│   ├── teacher.py            # Teacher API client (OpenAI-compatible)
│   └── scheduler/            # Graph-based execution scheduler
│       ├── graph.py          #   Computation-graph analysis
│       ├── cost.py           #   Operator cost model
│       ├── planner.py        #   Execution-plan construction
│       ├── buffers.py        #   Liveness-based buffer allocation
│       └── runtime.py        #   Execution scheduler (drop-in for model())
├── benchmarks/               # Kernel / model / parity benchmarks
│   ├── benchmark_*.py        # attention, block, cuda-graphs, exec-plan, …
│   ├── verify_*_parity.py    # Bit-parity checks vs the eager path
│   ├── profile_moe.py
│   └── results/              # Machine-specific measured reports (gitignored)
├── docs/                     # Design docs & reports (13)
│   ├── flash_attention.md    # FlashAttention-2 integration design & results
│   ├── kv_cache.md           # KV cache backends (static / quantized / MLA)
│   ├── exec_scheduler.md     # Graph-based execution scheduler
│   ├── moe_grouped_gemm.md   # Grouped-GEMM MoE engine
│   ├── packing.md            # Dynamic sequence packing design & benchmarks
│   ├── pipeline.md           # Overlapped training pipeline
│   ├── cuda_graphs.md        # CUDA-graph capture & replay
│   ├── expert_cache.md       # Persistent expert cache
│   ├── layer_prefetch.md     # Layer-expert prefetching
│   ├── fused_block.md        # Fused projections & RMSNorm
│   ├── mla.md                # Multi-head Latent Attention
│   ├── ARCHITECTURE_REVIEW.md
│   └── TRAINING_REPORT_100M.md
├── tests/                    # Pytest suite (run by CI)
│   ├── test_attn.py          # Attention dispatch & numerical equivalence
│   ├── test_kv.py            # KV backends (default / static / quantized / MLA)
│   ├── test_model.py         # Model forward shapes & fused projections
│   ├── test_generate.py      # Generation, sampling, chat
│   ├── test_moe.py           # MoE grouped-GEMM parity
│   ├── test_scheduler.py     # Execution-scheduler parity
│   ├── test_pipeline.py      # Overlapped pipeline stages
│   ├── test_cuda_graphs.py   test_expert_cache.py  test_layer_prefetch.py
│   ├── test_packing.py       test_distill.py  test_teacher.py
│   ├── test_config.py        test_data.py  test_cli.py  test_server.py
│   └── __init__.py
├── .github/workflows/        # CI pipeline
│   └── ci.yml
├── data/                     # Dataset generators + sample corpus
│   ├── generate_*.py         # cow / westbengal / bihar / cat / siraj datasets
│   └── sample.txt            # Small tracked corpus (notebook fallback)
├── Metis_Colab_Training.ipynb  # Colab GPU-training notebook
├── train_westbengal_100m.py  # 100M West-Bengal training (see docs/TRAINING_REPORT_100M.md)
├── start_tunnel.bat          # Expose the distill teacher gateway (Cloudflare tunnel)
├── stop_tunnel.bat
├── pyproject.toml            # Build config & metis CLI entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Git ignore rules
└── LICENSE                   # Unlicense (public domain)
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.4
- tqdm
- numpy

**Hardware:** Any machine will work. GPU with ≥2 GB VRAM recommended for faster training. CPU-only training is fully supported.

**FlashAttention-2:** no extra dependency is required — Metis uses PyTorch's
fused SDPA kernels (FA2 / memory-efficient) automatically. On Linux you may
optionally `pip install flash-attn` for the fastest kernel; Metis detects and
prefers it when present, and falls back to SDPA otherwise.

---

## License

This project is released into the public domain under the **Unlicense** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built with ❤️ as a learning project — from absolute scratch, no frameworks, no shortcuts.*

**Μῆτις** — *wisdom through craft*

</div>