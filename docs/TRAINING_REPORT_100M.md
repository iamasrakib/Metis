# Metis 100M — West Bengal Training Report

**Date**: 2026-08-03
**Model**: MetisLM 100M (99,172,608 parameters)
**Dataset**: West Bengal knowledge corpus
**Hardware**: CPU (no GPU available)

---

## 1. Dataset

| Metric | Value |
|--------|-------|
| Source | `data/westbengal.txt` |
| Characters | 36,153 |
| Words | 5,475 |
| Documents | 86 (paragraph-split) |
| Train/Val split | 77 / 9 documents (90%/10%) |
| Tokenizer | Character-level (`CharTokenizer`) |
| Vocab size | 74 (70 chars + 4 special tokens) |

**Content**: Comprehensive coverage of West Bengal including history, geography, culture, economy, politics, education, tourism, literature, festivals, cuisine, notable people, districts, architecture, sports, and environmental topics.

---

## 2. Architecture

| Parameter | Value |
|-----------|-------|
| Total parameters | 99,172,608 (99.2M) |
| Trainable parameters | 99,172,608 (100%) |
| Architecture | Decoder-only Transformer |
| d_model | 768 |
| n_heads | 16 |
| n_kv_heads | 4 (GQA, 4x KV compression) |
| n_layers | 16 |
| head_dim | 48 (768 / 16) |
| FFN type | SwiGLU |
| FFN hidden dim | 2,048 |
| Norm | RMSNorm (pre-norm) |
| Position | RoPE (Rotary Position Embeddings) |
| Attention | SDPA memory-efficient (auto) |
| Weight tying | Yes (embedding = LM head) |
| Max seq len | 256 |

### Parameter Breakdown (estimated)
- Embedding: 74 × 768 = 56,832
- Per layer: ~6.2M (QKV + O + SwiGLU w13/w2 + norms)
- 16 layers: ~99.1M
- Final norm: 768

---

## 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (fused=False on CPU) |
| Learning rate | 3e-4 (peak) |
| Min learning rate | 1e-5 |
| LR schedule | Cosine decay with linear warmup |
| Warmup steps | 5 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Micro batch size | 2 |
| Gradient accumulation | 8 |
| Effective batch size | 16 |
| Max iterations | 30 (demo run) |
| AMP | Disabled (CPU) |
| Gradient checkpointing | Enabled |
| CUDA Graphs | Disabled (required for grad ckpt) |
| Overlapped pipeline | Enabled |
| Dynamic packing | Enabled (stream strategy) |
| Seed | 42 |

---

## 4. Training Results

### Loss Curve

| Step | Loss | LR | Grad Norm | Speed |
|------|------|----|-----------|-------|
| 0 | 4.4485 | 6.00e-05 | 37.05 | 0.1 steps/s |
| 5 | 3.7706 | 3.00e-04 | 7.19 | 0.1 steps/s |
| 10 | 3.1822 | 2.72e-04 | 2.16 | 0.1 steps/s |
| 15 | 3.1355 | 2.00e-04 | 1.10 | 0.1 steps/s |
| 20 | 3.1132 | 1.10e-04 | 0.74 | 0.1 steps/s |
| 25 | 3.0535 | 3.77e-05 | 0.65 | 0.1 steps/s |
| 29 | 2.9917 | 1.11e-05 | 0.51 | 0.1 steps/s |

### Summary

| Metric | Value |
|--------|-------|
| Final loss | 3.3043 (avg of last 100) |
| Final perplexity | 27.23 |
| Final step loss | 2.9917 |
| Loss reduction | 4.45 → 2.99 (33% reduction) |
| Wall time | 545.2s (9.1 min) |
| Steps/sec | 0.055 (CPU, 100M params) |
| Time per step | ~18.1s (CPU) |

### Observations

1. **Loss decreased consistently** from 4.45 to 2.99 over 30 steps, showing the model is learning.
2. **Gradient norms decreased** from 37.05 to 0.51, indicating training stabilization.
3. **Sample outputs** at step 10 and 20 show the model is beginning to learn character patterns but hasn't converged (expected with only 30 steps on CPU).
4. **No NaNs or numerical instability** — training was stable throughout.
5. **CPU performance**: ~18 seconds per step for 100M params is expected. On an RTX 2050 with AMP, this would be ~0.5-1s per step.

---

## 5. Checkpoints

| File | Size | Description |
|------|------|-------------|
| `best_model.pt` | 383 MB | Best model weights (seeded from final since no val) |
| `final_model.pt` | 383 MB | Final training state |
| `latest_checkpoint.pt` | 1.2 GB | Full checkpoint (model + optimizer + config) |
| `config.json` | 1.9 KB | ModelConfig JSON |
| `tokenizer.json` | 2.3 KB | CharTokenizer state |
| `training_meta.json` | 497 B | Training metadata |
| **Total** | **~1.9 GB** | |

---

## 6. Performance Notes

### CPU vs GPU

| Metric | CPU (this run) | RTX 2050 (estimated) |
|--------|---------------|---------------------|
| Time/step | ~18s | ~0.5-1s |
| 30 steps | 9.1 min | ~15-30 sec |
| 500 steps | ~2.5 hours | ~4-8 min |
| Throughput | ~0.055 steps/s | ~1-2 steps/s |

### VRAM Estimate (RTX 2050 4GB)

| Component | Size |
|-----------|------|
| Model weights (fp32) | 383 MB |
| Optimizer states (AdamW) | ~766 MB |
| Gradients | ~383 MB |
| Activations (grad ckpt) | ~200 MB |
| **Total** | **~1.7 GB** |

The 100M model fits comfortably in 4GB VRAM with gradient checkpointing.

---

## 7. What Would a Full Training Run Look Like?

For production-quality results on an RTX 2050:

```bash
# Recommended full training configuration
# (modify max_iters to 500-2000)
python train_westbengal_100m.py
```

| Setting | Demo (this run) | Recommended Full |
|---------|----------------|-----------------|
| max_iters | 30 | 500-2000 |
| max_seq_len | 256 | 256-512 |
| gradient_accumulation | 8 | 4-8 |
| warmup_steps | 5 | 50-100 |
| val_interval | 100 | 100 |
| Expected loss | 2.99 | <2.0 |
| Expected perplexity | 27.2 | <7.5 |
| Training time (GPU) | 15s | 5-20 min |

---

## 8. Files Generated

```
data/
  westbengal.txt                    # 36 KB dataset
  generate_westbengal_dataset.py    # Dataset generator

checkpoints_westbengal_100m/
  best_model.pt                     # Best weights (383 MB)
  final_model.pt                    # Final weights (383 MB)
  latest_checkpoint.pt              # Full checkpoint (1.2 GB)
  config.json                       # ModelConfig
  tokenizer.json                    # CharTokenizer
  training_meta.json                # Training metadata

train_westbengal_100m.py           # Training script
docs/TRAINING_REPORT_100M.md       # This report
```
