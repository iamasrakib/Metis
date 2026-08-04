#!/usr/bin/env python
"""
Train a 100M parameter Metis model on the combined West Bengal + Bihar dataset.

The combined corpus (data/combined.txt) is built from data/westbengal.txt and
data/bihar.txt — the script auto-builds it from those two files if it does not
already exist, so this works standalone or from the Colab notebook.

Architecture: same as the West Bengal run —
  d_model=768, n_heads=16, n_kv_heads=4 (GQA), n_layers=16, SwiGLU, max_seq_len=256
"""
import os
import sys
import time
import math
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from metis.config import ModelConfig, setup_logging
from metis.data import CharTokenizer, load_text, train_val_split, create_dataloader
from metis.model import MetisLM
from metis.training import train

# ── Combined dataset ─────────────────────────────────────────────────────────
# Auto-build data/combined.txt from the two state corpora if it's missing.
SOURCE_FILES = ["data/westbengal.txt", "data/bihar.txt"]
COMBINED_PATH = "data/combined.txt"


def ensure_combined_data() -> None:
    if os.path.exists(COMBINED_PATH):
        return
    parts = []
    for src in SOURCE_FILES:
        if not os.path.exists(src):
            print(f"ERROR: missing dataset {src} — run the dataset generator first.")
            sys.exit(1)
        with open(src, encoding="utf-8") as f:
            parts.append(f.read().strip())
    with open(COMBINED_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
    print(f"Combined dataset created: {COMBINED_PATH}")


# ── Configuration ────────────────────────────────────────────────────────────
# 100M parameter configuration
config = ModelConfig(
    # Architecture
    d_model=768,
    n_heads=16,
    n_kv_heads=4,       # GQA: 4 query groups -> 16/4 = 4x KV compression
    n_layers=16,
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
    micro_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch = 16
    max_iters=1000,             # Bigger corpus => more steps; change freely
    learning_rate=3e-4,
    min_lr=1e-5,
    weight_decay=0.1,
    warmup_steps=5,
    max_grad_norm=1.0,
    val_interval=100,           # Disable val for short run
    val_steps=20,
    sample_interval=10,         # Sample every 10 steps
    save_interval=10,           # Checkpoint every 10 steps
    log_interval=1,
    train_split=0.9,

    # Data
    dataset_path=COMBINED_PATH,
    checkpoint_dir="checkpoints_combined_100m",
    tokenizer="char",
    use_mmap=True,
    num_workers=0,
    use_packing=True,          # Zero-padding waste
    packing_strategy="stream", # Contiguous packing

    # Runtime — auto-detect; uses gradient checkpointing when no CUDA graphs
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    compile_model=False,
    use_cuda_graphs=False,     # OFF: enables gradient checkpointing
    use_pipeline=True,
    prefetch_depth=2,
    async_checkpoint=True,
    pipeline_buffer_depth=3,
    use_layer_prefetch=False,  # OFF: no MoE in this config

    # Seed
    seed=42,
    log_level="INFO",
)


def main():
    print("=" * 70)
    print("  Metis 100M — West Bengal + Bihar Combined Training Run")
    print("=" * 70)

    ensure_combined_data()

    # Validate dataset exists
    if not os.path.exists(config.dataset_path):
        print(f"ERROR: Dataset not found at {config.dataset_path}")
        sys.exit(1)

    # Quick sanity: load text to check size
    text = load_text(config.dataset_path)
    n_chars = len(text)
    n_words = len(text.split())
    print(f"\n  Dataset: {config.dataset_path}")
    print(f"  Characters: {n_chars:,}")
    print(f"  Words: {n_words:,}")

    # Tokenize to get vocab size
    tok = CharTokenizer()
    tok.fit(text)
    config.vocab_size = tok.vocab_size
    print(f"  Vocab size: {config.vocab_size}")

    # Build model to get param count
    model = MetisLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Architecture:")
    print(f"    d_model:      {config.d_model}")
    print(f"    n_heads:      {config.n_heads}")
    print(f"    n_kv_heads:   {config.n_kv_heads} (GQA groups: {config.n_heads // config.n_kv_heads})")
    print(f"    n_layers:     {config.n_layers}")
    print(f"    max_seq_len:  {config.max_seq_len}")
    print(f"    FFN:          SwiGLU (hidden={model.layers[0].ffn.hidden})")
    print(f"    Total params: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"    Trainable:    {n_trainable:,}")

    # Hardware info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print(f"\n  Device: CPU (no CUDA)")

    print(f"\n  Training:")
    print(f"    Iterations:   {config.max_iters}")
    print(f"    Batch size:   {config.micro_batch_size} × {config.gradient_accumulation_steps} = {config.effective_batch_size}")
    print(f"    Learning rate: {config.learning_rate}")
    print(f"    Warmup steps: {config.warmup_steps}")
    print(f"    CUDA Graphs:  {'ON' if config.use_cuda_graphs else 'OFF (grad checkpointing ON)'}")
    print(f"    Pipeline:     {'ON' if config.use_pipeline else 'OFF'}")
    print(f"    Packing:      {config.packing_strategy if config.use_packing else 'OFF'}")
    print(f"    Device:       {config.device}")
    print("=" * 70)

    # Run training
    t0 = time.time()
    train(config, resume=False)
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Checkpoints saved to: {config.checkpoint_dir}/")
    print(f"{'=' * 70}")

    # Save training metadata
    meta = {
        "model": "MetisLM 100M (combined WB + Bihar)",
        "dataset": config.dataset_path,
        "dataset_chars": n_chars,
        "dataset_words": n_words,
        "vocab_size": config.vocab_size,
        "n_params": n_params,
        "architecture": {
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "n_layers": config.n_layers,
            "max_seq_len": config.max_seq_len,
            "ffn_hidden": model.layers[0].ffn.hidden,
        },
        "training": {
            "max_iters": config.max_iters,
            "effective_batch_size": config.effective_batch_size,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
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
