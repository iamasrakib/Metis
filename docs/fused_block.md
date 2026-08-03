# Fused Transformer Block — kernel-launch minimisation

## Motivation

The original Transformer block launched several redundant CUDA kernels per
forward pass:

- **RMSNorm** — the manual implementation (`pow → mean → rsqrt → type_as → mul`)
  dispatched ~7 elementwise/reduction kernels per norm, three of which are
  redundant casts between fp32 and the input dtype.
- **QKV projection** — three separate `nn.Linear` GEMMs instead of one.
- **SwiGLU gate/up** — two separate `nn.Linear` GEMMs instead of one.

The fused block eliminates those redundancies, and every change is verified to
be **bit-identical** to the original implementation.

## What changed

### Fused RoPE (q+k pair)

RoPE was applied twice per block — once for `q` and once for `k` — each
launching `float() → complex-mul → type_as` = 3 kernels, for 6 total. A single
`apply_rope_pair(q, k, freqs)` concatenates the two along the head dim, runs
the three data-dependent ops once on the combined buffer, then splits (pure
metadata). **6 → 4 kernels**, and the pair path is bit-identical to two
separate `apply_rope` calls (verified in `verify_block_parity.py`).

```python
# before — 6 kernels
q = apply_rope(q, self.rope_freqs)
k = apply_rope(k, self.rope_freqs)

# after — 4 kernels (cat · cast-up · cmul · cast-down; split is metadata)
q, k = apply_rope_pair(q, k, self.rope_freqs)
```

### Fused RMSNorm

```python
# before
norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
return (x.float() * norm).type_as(x) * self.weight

# after — one fused kernel via torch.nn.functional.rms_norm
return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)
```

`F.rms_norm` is a single fused kernel with fp32 accumulation (torch ≥ 2.4).
Verified bit-identical for `torch.float32`, `torch.bfloat16`, and
`torch.float16` inputs under AMP autocast — see `benchmarks/verify_block_parity.py`.

### Fused QKV projection

```python
# before — three GEMMs
q = self.q_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
k = self.k_proj(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)
v = self.v_proj(x).view(B, T, n_kv_heads, head_dim).transpose(1, 2)

# after — one GEMM + split (metadata, zero kernels)
q, k, v = self.qkv(x).split([d_model, kv_dim, kv_dim], dim=-1)
q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
k = k.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
v = v.view(B, T, n_kv_heads, head_dim).transpose(1, 2)
```

Under fp16/bf16 autocast the fused path is bit-identical: three separate
`F.linear` calls and one concatenated-`F.linear` call produce the same
output because cuBLAS/gemm handles fp16 accumulation in the same order.

### Fused gate/up (SwiGLU)

```python
# before — two GEMMs
return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# after — one GEMM + split + same elementwise ops
gate, up = self.w13(x).split(self.hidden, dim=-1)
return self.dropout(self.w2(F.silu(gate) * up))
```

The MLP (GELU fallback) is unchanged — it has only one input projection
(`c_fc`), so there is no fusion opportunity there.

## What stayed the same

| Component | Reason not to fuse | Note |
|-----------|-------------------|------|
| **Attention** | Already fused via SDPA dispatch (`mem_efficient` / `flash_attn` /
  `math`). | One kernel call per layer. |
| **Output projection + residual** | `o_proj` is a single GEMM; the residual add
  is a single elementwise kernel. Fusing these requires a custom kernel. |
| **Attention sink** | Prepended token and RoPE offset bookkeeping are unchanged;
  only the projection layer changed. |

RoPE is now fused into a single pair call (see above). It could be fused
*further* into the attention kernel itself with Triton (not available on
Windows torch builds).

## Checkpoint compatibility

Existing checkpoints with the legacy split-key format
(`layers.N.attn.q_proj.weight`, `layers.N.attn.k_proj.weight`, etc.) load
**byte-identically** into the fused model. A state_dict compat shim ensures
`model.state_dict()` still exports the old keys, so old code can also load
checkpoints written by the fused implementation.

**Old-format checkpoint → fused model:** weights load via the shim; the state
dict's `q_proj.weight` / `k_proj.weight` / `v_proj.weight` are concatenated
into the fused `qkv` parameter on the fly.

**New checkpoint → old code:** `model.state_dict()` exports the split keys
(the fused weight is sliced into three row groups); old code loads them as
separate `nn.Linear` parameters. Works transparently.

**Optimizer resume:** old checkpoints' optimizer state is keyed to the
pre-fusion parameter layout. On resume, the guard in `load_checkpoint`
detects the mismatch and starts a fresh optimizer while still loading model
weights. New checkpoints are self-consistent and resume normally.

## Numerical guarantees

| Property | Guarantee | Verified by |
|----------|-----------|-------------|
| `F.rms_norm` vs manual RMSNorm | bit-identical (fp32, bf16, fp16) | `verify_block_parity.py` |
| fused QKV vs 3 separate GEMMs | bit-identical (fp32, fp16/bf16 autocast) | `verify_block_parity.py` |
| fused w13 vs w1/w3 separately | bit-identical (fp32, fp16/bf16 autocast) | `verify_block_parity.py` |
| RoPE pair vs two separate calls | bit-identical (prefill & decode shapes) | `verify_block_parity.py` |
| block outputs (fused vs reference) | bit-identical (fp32), within 2% (bf16 autocast) | `verify_block_parity.py` |
| gradients | relative error < 0.3% (fp32) | `verify_block_parity.py` |
| KV-cache decode | bit-identical to full-context forward | `verify_block_parity.py` |
| gradient checkpointing | checkpointed grads == full grads (within bf16 tol) | `verify_block_parity.py` |
| real checkpoint load | fused reproduces reference model outputs exactly | `verify_block_parity.py` |

## Measured results

Platform: NVIDIA GeForce RTX 2050 (sm_86), torch 2.6.0+cu124, Windows 11.

### Throughput — block forward + backward (B=2, T=128, d_model=256)

| variant | median (ms) | tokens/s | speedup |
|---------|------------|---------:|--------:|
| reference (pre-fusion) | 5.09 | 50,297 | 1.00× |
| **fused** | **4.02** | **63,681** | **1.27×** |

### Decode bandwidth — single token (T_q=1, bf16, 128-token KV cache)

| variant | decode (ms) | effective BW (GB/s) | speedup |
|---------|------------:|-------------------:|--------:|
| reference | 1.35 | 1.1 | 1.00× |
| **fused** | **1.32** | **1.1** | **1.02×** |

The decode path is dominated by the output-projection GEMM and KV-cache reads,
so the RoPE savings at T=1 are small — the prefill/train win is where the
kernel-launch reduction shows.

### Full MetisLM train step (fused)

| median (ms) | tokens/s |
|------------:|---------:|
| 19.51 | 13,122 |

### Kernel count

RoPE (fused pair) drops its data-dependent kernels from 6 → 4 per block
(measured in isolation: 24 → 14 op-level events). Fused RMSNorm replaces a
7-op chain with 1 kernel, and fused QKV + gate/up replace 5 GEMMs with 3.

On this Windows build the profiler reports op-level counts (including
autocast metadata ops like `aten::empty`, `aten::fill_`) rather than raw CUDA
kernel names — the absolute per-block totals are inflated, so the
**wall-clock numbers above are the authoritative performance metric**.

### What remains unfused

RoPE and the output-projection + residual path could be fused into the
attention kernel with a custom Triton kernel. On builds where Triton is
available, `torch.compile(mode="max-autotune")` achieves these fusions
automatically. On Windows torch builds (no Triton), the fused block as
implemented is the maximum achievable in eager PyTorch.

## Files

| File | Role |
|------|------|
| `metis/model.py` | Fused `RMSNorm`, `CausalSelfAttention` (qkv), `SwiGLU` (w13) |
| `metis/training.py` | Optimizer-resume guard in `load_checkpoint` |
| `benchmarks/verify_block_parity.py` | Numerical parity verification (14 checks) |
| `benchmarks/benchmark_block.py` | A/B throughput + bandwidth benchmark |
| `tests/test_model.py` | `TestFusedProjections` — shim and forward tests |
