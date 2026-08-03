#!/usr/bin/env python
"""
Train a SMALL Metis model on the West Bengal dataset — fast enough for CPU.

Why small? The 100M model is ~30x more capacity than a 36K-char dataset can
fill, so it never learns word order. A ~5.7M char-level model CAN memorize and
reproduce this corpus, producing readable (if not conversational) text after
a few thousand steps — on CPU in under an hour.

Architecture:
  d_model=256, n_heads=8, n_kv_heads=2 (GQA), n_layers=6, max_seq_len=256
  SwiGLU FFN (hidden≈1024), ~5.7M params
"""
import os
import sys
import time
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from metis.config import ModelConfig
from metis.data import CharTokenizer, load_text, train_val_split, create_dataloader
from metis.model import MetisLM
from metis.training import train

# ── Configuration ────────────────────────────────────────────────────────────
config = ModelConfig(
    # Architecture
    d_model=256,
    n_heads=8,
    n_kv_heads=2,       # GQA: 8/2 = 4x KV compression
    n_layers=6,
    max_seq_len=256,
    dropout=0.1,
    use_rmsnorm=True,
    use_swiglu=True,
    use_rope=True,
    tie_weights=True,
    use_moe=False,
    use_qk_norm=False,
    use_attention_sink=False,

    # Training
    micro_batch_size=8,          # ≤ val-set packed sequences (12) or packing rejects the config
    gradient_accumulation_steps=4,   # effective batch = 32
    max_iters=2000,              # plenty of passes over the tiny corpus
    learning_rate=1e-3,
    min_lr=1e-4,
    weight_decay=0.1,
    warmup_steps=100,
    max_grad_norm=1.0,
    val_interval=100,           # validate every 100 steps
    val_steps=20,
    sample_interval=100,        # watch it learn
    save_interval=500,
    log_interval=10,
    train_split=0.9,

    # Data
    dataset_path="data/westbengal.txt",
    checkpoint_dir="checkpoints_westbengal_small",
    tokenizer="char",
    use_mmap=True,
    num_workers=0,
    use_packing=True,
    packing_strategy="stream",

    # Runtime
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    compile_model=False,
    use_cuda_graphs=False,
    use_pipeline=True,
    prefetch_depth=2,
    async_checkpoint=True,
    pipeline_buffer_depth=3,
    use_layer_prefetch=False,

    # Seed
    seed=42,
    log_level="INFO",
)


def main():
    print("=" * 70)
    print("  Metis SMALL — West Bengal Training Run")
    print("=" * 70)

    if not os.path.exists(config.dataset_path):
        print(f"ERROR: Dataset not found at {config.dataset_path}")
        sys.exit(1)

    text = load_text(config.dataset_path)
    print(f"\n  Dataset: {config.dataset_path} ({len(text):,} chars)")

    tok = CharTokenizer()
    tok.fit(text)
    config.vocab_size = tok.vocab_size
    print(f"  Vocab size: {config.vocab_size}")

    model = MetisLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Architecture: d_model={config.d_model} h={config.n_heads} "
          f"kv={config.n_kv_heads} L={config.n_layers}")
    print(f"  Total params: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Device:       {config.device}")
    print(f"  Iterations:   {config.max_iters} "
          f"(batch {config.micro_batch_size} x {config.gradient_accumulation_steps} "
          f"= {config.effective_batch_size})")
    print("=" * 70)

    t0 = time.time()
    train(config, resume=False)
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Checkpoints saved to: {config.checkpoint_dir}/")
    print(f"{'=' * 70}")

    meta = {
        "model": "MetisLM SMALL",
        "dataset": config.dataset_path,
        "dataset_chars": len(text),
        "vocab_size": config.vocab_size,
        "n_params": n_params,
        "architecture": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "n_layers": config.n_layers,
            "max_seq_len": config.max_seq_len,
        },
        "training": {
            "max_iters": config.max_iters,
            "effective_batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
        },
        "wall_time_sec": elapsed,
    }
    meta_path = os.path.join(config.checkpoint_dir, "training_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")


if __name__ == "__main__":
    import torch
    main()
