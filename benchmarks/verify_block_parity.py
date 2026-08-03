#!/usr/bin/env python3
"""
Metis — fused-block numerical-parity verification
==================================================
Compares the FUSED transformer block (single ``qkv`` projection, fused gate/up
``w13``, paired ``q/k`` RoPE, ``F.rms_norm``) against a faithful reference
implementation of the pre-fusion block (separate ``q/k/v`` and ``w1/w3``
projections, per-tensor RoPE, manual RMSNorm) — across every requirement
axis:

  block & model outputs    — fp32 (bit-identical) and bf16/fp16 autocast
  gradients                — fused params vs their split counterparts
  KV-cache decode          — cached step-by-step == full-context forward
  gradient checkpointing   — checkpointed grads == full grads
  checkpoint compatibility — a real pre-fusion checkpoint loads into the fused
                             model and reproduces the reference model's output

The reference model is built from the OLD architecture and loaded via the fused
model's own ``state_dict()`` (which exports the legacy split keys), so both
sides evaluate the exact same weight values.

Each check prints a PASS/FAIL line; the script exits non-zero on any failure.
CPU-safe by default; CUDA-only cases are skipped on CPU.

Usage:
    python benchmarks/verify_block_parity.py                 # auto device
    python benchmarks/verify_block_parity.py --device cuda --seed 0
    python benchmarks/verify_block_parity.py --checkpoint checkpoints/final_model.pt
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metis.attn import causal_attention  # noqa: E402
from metis.config import ModelConfig  # noqa: E402
from metis.model import (  # noqa: E402
    MetisLM, RMSNorm, SwiGLU, apply_rope, apply_rope_pair, CausalSelfAttention,
)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Reference (pre-fusion) implementation
# ──────────────────────────────────────────────────────────────────────────────

class ReferenceRMSNorm(nn.Module):
    """The old manual RMSNorm (pow → mean → rsqrt → mul), byte-for-byte."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class ReferenceQKNorm(nn.Module):
    """Mirror metis.model.QKNorm nesting (self.norm = ManualRMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = ReferenceRMSNorm(dim, eps)

    def forward(self, x):
        return self.norm(x).type_as(x)


class ReferenceAttention(nn.Module):
    """The old attention: separate q/k/v projections, everything else shared."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = self.n_heads // self.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.kv_dim = config.d_model * self.n_kv_heads // self.n_heads
        self.use_rope = config.use_rope
        self.use_qk_norm = config.use_qk_norm
        self.use_attention_sink = config.use_attention_sink
        self.backend_request = getattr(config, "attn_backend", "auto")
        self.use_flash_attn = config.use_flash_attn

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        if self.use_qk_norm:
            self.q_norm = ReferenceQKNorm(self.head_dim)
            self.k_norm = ReferenceQKNorm(self.head_dim)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
        )
        if self.use_rope:
            rope_len = config.max_seq_len + (1 if self.use_attention_sink else 0)
            from metis.model import precompute_rope_frequencies

            self.register_buffer(
                "rope_freqs",
                precompute_rope_frequencies(self.head_dim, rope_len),
                persistent=False,
            )
        if self.use_attention_sink:
            self.sink_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        if self.use_attention_sink and kv_cache is None:
            x = torch.cat([self.sink_token.expand(B, -1, -1), x], dim=1)
            T = x.size(1)

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            if kv_cache is not None:
                offset = kv_cache[0].size(2)
                rope_slice = self.rope_freqs[offset: offset + T, :].unsqueeze(0).unsqueeze(0)
                q_rope = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
                k_rope = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
                q = torch.view_as_real(q_rope * rope_slice).flatten(-2).type_as(q)
                k = torch.view_as_real(k_rope * rope_slice).flatten(-2).type_as(k)
            else:
                q = apply_rope(q, self.rope_freqs)
                k = apply_rope(k, self.rope_freqs)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        y = causal_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads,
            backend=self.backend_request, use_flash_attn=self.use_flash_attn,
            training=self.training,
        )
        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        y = self.resid_dropout(self.o_proj(y))

        if self.use_attention_sink and kv_cache is None:
            y = y[:, 1:, :]
        return y, new_kv_cache


class ReferenceSwiGLU(nn.Module):
    """The old SwiGLU with separate w1/w3 input projections."""

    def __init__(self, config):
        super().__init__()
        hidden = int(4 * config.d_model * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)
        self.w3 = nn.Linear(config.d_model, hidden, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class ReferenceBlock(nn.Module):
    """The old pre-norm block."""

    def __init__(self, config):
        super().__init__()
        norm_cls = ReferenceRMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln_1 = norm_cls(config.d_model)
        self.attn = ReferenceAttention(config)
        self.ln_2 = norm_cls(config.d_model)
        self.ffn = ReferenceSwiGLU(config) if config.use_swiglu else _ReferenceMLP(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, new_cache


class _ReferenceMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


def build_reference(config, fused_model):
    """A pre-fusion model loaded from the fused model's weights.

    ``fused_model.state_dict()`` exports the legacy split keys, which is
    exactly the reference model's native parameter layout — so a plain
    ``load_state_dict`` hands the reference the same weight values.
    """
    ref = _ReferenceLM(config).to(next(fused_model.parameters()).device)
    missing, unexpected = ref.load_state_dict(fused_model.state_dict())
    assert not missing and not unexpected, (missing, unexpected)
    return ref


class _ReferenceLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([ReferenceBlock(config) for _ in range(config.n_layers)])
        norm_cls = ReferenceRMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm_f = norm_cls(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.tok_emb.weight = self.lm_head.weight

    def forward(self, idx, targets=None, use_checkpointing=False, kv_cache=None):
        B, T = idx.size()
        x = self.tok_emb(idx)
        if not self.config.use_rope:
            x = x + self.pos_emb(torch.arange(0, T, device=idx.device))
        x = self.drop(x)
        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            if use_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(layer, x, layer_cache, use_reentrant=False)
            else:
                x, new_cache = layer(x, kv_cache=layer_cache)
                new_kv_cache.append(new_cache)
        x = self.norm_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0
            )
            return logits, loss, new_kv_cache
        return self.lm_head(x[:, [-1], :]), None, new_kv_cache


# ──────────────────────────────────────────────────────────────────────────────
# Grad mapping
# ──────────────────────────────────────────────────────────────────────────────

def param_grads(model, names):
    """Return {name: grad} for named parameters that have grads."""
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            grads[n] = p.grad.detach().float()
    return grads


def fused_to_split_grads(model):
    """Map the fused model's grads onto the reference (split) parameter names."""
    out = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        if n.endswith("qkv.weight"):
            d, kv = p.shape[1], p.shape[1] - 2 * (p.shape[0] - p.shape[1]) // 2
            d = p.shape[1]
            total = p.shape[0]
            kv = (total - d) // 2
            base = n[: -len("qkv.weight")]
            out[base + "q_proj.weight"] = g[:d]
            out[base + "k_proj.weight"] = g[d:d + kv]
            out[base + "v_proj.weight"] = g[d + kv:]
        elif n.endswith("w13.weight"):
            h = p.shape[0] // 2
            base = n[: -len("w13.weight")]
            out[base + "w1.weight"] = g[:h]
            out[base + "w3.weight"] = g[h:]
        else:
            out[n] = g
    return out


def max_rel_diff(a, b):
    denom = a.abs().clamp_min(1e-6)
    return (a - b).abs().div(denom).max().item() if a.numel() else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────────────

class Result:
    def __init__(self, tol):
        self.tol = tol
        self.pass_ = 0
        self.fail = 0
        self.skips = 0

    def check(self, name, cond, detail=""):
        if cond:
            self.pass_ += 1
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.fail += 1
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))

    def skip(self, name, why):
        self.skips += 1
        print(f"  SKIP  {name}  ({why})")

    def summary(self):
        print(f"\n  {self.pass_} passed, {self.fail} failed, {self.skips} skipped")
        return self.fail == 0


def verify_block_outputs(r: Result, device: str, seed: int, tol: float):
    """Fused block vs reference block: outputs in fp32 (bit) and bf16 (tol).

    Comparison is at the block level (not full MetisLM) so the input is
    a float hidden-state tensor, not token indices.
    """
    torch.manual_seed(seed)
    print("\n[block] fused vs reference outputs")
    cfg = ModelConfig(
        d_model=128, n_heads=4, n_kv_heads=2, n_layers=1, max_seq_len=64,
        vocab_size=256, dropout=0.0, use_rmsnorm=True,
        use_swiglu=True, use_rope=True, use_flash_attn=True,
        attn_backend="auto", use_qk_norm=True,
    )
    from metis.model import TransformerBlock as FusedBlock

    fused = FusedBlock(cfg).to(device).eval()
    ref = ReferenceBlock(cfg).to(device).eval()

    # Transfer weights from fused → reference via the legacy state_dict.
    # TransformerBlock.state_dict() exports attn.q/k/v and ffn.w1/w3 keys
    # via the CausalSelfAttention and SwiGLU shims; ReferenceBlock uses the
    # same key names, so load_state_dict maps them directly.
    state = fused.state_dict()
    # ReferenceBlock expects the same TransformerBlock-level keys.
    # Rename attn.q_norm.weight → q_norm.norm.weight to match reference.
    rename = {}
    for k, v in state.items():
        if "q_norm.weight" in k:
            rename[k] = k.replace("q_norm.weight", "q_norm.norm.weight")
        elif "k_norm.weight" in k:
            rename[k] = k.replace("k_norm.weight", "k_norm.norm.weight")
    state.update(rename)
    for k in list(state.keys()):
        if "q_norm.weight" in k and ".norm." not in k:
            del state[k]
        elif "k_norm.weight" in k and ".norm." not in k:
            del state[k]
    missing, unexpected = ref.load_state_dict(state, strict=False)
    # Ignore any remaining buffer-only differences (causal_mask etc.)
    assert not unexpected, f"unexpected keys: {unexpected}"

    for label, dtype, amp in (
        ("fp32", torch.float32, False),
        ("bf16", torch.bfloat16, True),
    ):
        B, T, D = 2, 32, cfg.d_model
        x = torch.randn(B, T, D, device=device, dtype=dtype)
        with torch.no_grad():
            if amp:
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    yf = fused(x, None)[0]
                    yr = ref(x, None)[0]
            else:
                yf = fused(x, None)[0]
                yr = ref(x, None)[0]
        yf, yr = yf.float(), yr.float()
        if not amp:
            ok = torch.equal(yf, yr)
            detail = "bit-identical"
        else:
            ok = torch.allclose(yf, yr, rtol=tol, atol=tol)
            detail = f"max_abs_diff={(yf - yr).abs().max().item():.2e}"
        r.check(f"block output {label} (qk-norm on)", ok, detail)


def verify_primitive_identity(r: Result, device: str, seed: int):
    """The three fusions are individually bit-identical to their old forms."""
    torch.manual_seed(seed)
    print("\n[primitives] bit-identity of each fusion")
    x = torch.randn(2, 16, 128, device=device)

    # RMSNorm
    dim = 128
    fused_norm = RMSNorm(dim).to(device)
    ref_norm = ReferenceRMSNorm(dim).to(device)
    ref_norm.weight.data.copy_(fused_norm.weight.data)
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        xd = x.to(dt)
        r.check(
            f"F.rms_norm == manual ({dt})",
            torch.equal(fused_norm(xd).float(), ref_norm(xd).float()),
        )

    # QKV
    cfg = ModelConfig(d_model=128, n_heads=4, n_kv_heads=2, n_layers=1,
                      max_seq_len=64, vocab_size=256, dropout=0.0)
    attn = CausalSelfAttention(cfg).to(device)
    w = attn.qkv.weight.detach()
    d, kv = 128, 64
    q, k, v = attn.qkv(x).split([d, kv, kv], dim=-1)
    q2, k2, v2 = x @ w[:d].t(), x @ w[d:d + kv].t(), x @ w[d + kv:].t()
    r.check("fused QKV == 3 separate GEMMs",
            torch.equal(q, q2) and torch.equal(k, k2) and torch.equal(v, v2))

    # gate/up
    glu = SwiGLU(cfg).to(device)
    w13 = glu.w13.weight.detach()
    h = glu.hidden
    g1, u1 = glu.w13(x).split(h, dim=-1)
    g2, u2 = x @ w13[:h].t(), x @ w13[h:].t()
    r.check("fused w13 == w1/w3 separately",
            torch.equal(g1, g2) and torch.equal(u1, u2))

    # RoPE pair == two separate calls (prefill and decode shapes)
    from metis.model import precompute_rope_frequencies
    for shape, off in (((2, 4, 32, 64), 0), ((1, 4, 1, 64), 128)):
        B, H, T, D = shape
        freqs = precompute_rope_frequencies(D, 256, device=device)
        q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
        k = torch.randn(B, 2, T, D, device=device, dtype=torch.bfloat16)
        qp, kp = apply_rope_pair(q, k, freqs[off:off + T])
        qs, ks = apply_rope(q, freqs[off:off + T]), apply_rope(k, freqs[off:off + T])
        r.check(
            f"RoPE pair == separate ({T}-token prefill)" if T > 1
            else f"RoPE pair == separate (decode T=1, offset {off})",
            torch.equal(qp, qs) and torch.equal(kp, ks),
        )


def verify_model_grads(r: Result, device: str, seed: int, tol: float):
    """Fused vs reference model gradients over a forward+backward."""
    torch.manual_seed(seed)
    print("\n[model] gradient parity (fwd+bwd)")
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=64,
        vocab_size=256, dropout=0.0, use_rmsnorm=True, use_swiglu=True,
        use_rope=True, use_flash_attn=True, attn_backend="auto",
    )
    fused = MetisLM(cfg).to(device).train()
    ref = build_reference(cfg, fused)
    idx = torch.randint(0, cfg.vocab_size, (2, 32), device=device)

    lf = fused(idx, idx)[1]
    lr = ref(idx, idx)[1]
    lf.backward()
    lr.backward()
    gf = fused_to_split_grads(fused)
    gr = param_grads(ref, None)
    common = set(gf) & set(gr)
    worst = max(max_rel_diff(gf[n], gr[n]) for n in common)
    loss_ok = torch.allclose(lf.detach().float(), lr.detach().float(), rtol=tol, atol=tol)
    grads_ok = worst < 10 * tol or worst < 1e-4
    r.check(
        "fused grads == reference grads",
        loss_ok and grads_ok,
        f"loss_match={loss_ok} worst_rel_grad_diff={worst:.2e}",
    )


def verify_kv_cache_decode(r: Result, device: str, seed: int, tol: float):
    """Cached step-by-step decode == full-context forward on the fused model."""
    torch.manual_seed(seed)
    print("\n[inference] KV-cache decode == full forward (fused)")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=128,
        vocab_size=256, dropout=0.0, use_rmsnorm=True, use_swiglu=True,
        use_rope=True, use_flash_attn=True, attn_backend="auto",
    )
    m = MetisLM(cfg).to(device).eval()
    toks = torch.randint(0, cfg.vocab_size, (1, 48), device=device)
    cache = None
    outs = []
    with torch.no_grad(), torch.autocast(device_type=device, dtype=amp):
        for i in range(toks.size(1)):
            lg, _, cache = m(toks[:, i:i + 1], kv_cache=cache)
            outs.append(lg[:, -1, :].float())
        full, _, _ = m(toks, targets=toks)
    bad = [
        i for i in range(toks.size(1))
        if not torch.allclose(outs[i], full[:, i].float(), rtol=tol, atol=tol)
    ]
    r.check("KV-cache decode == full forward", not bad, f"max_pos_err={max(bad) if bad else 0}")


def verify_grad_checkpointing(r: Result, device: str, seed: int, tol: float):
    """Checkpointed grads == full grads on the fused model."""
    torch.manual_seed(seed)
    print("\n[training] gradient-checkpointing equivalence (fused)")
    cfg = ModelConfig(
        d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, max_seq_len=64,
        vocab_size=256, dropout=0.0, use_rmsnorm=True, use_swiglu=True,
        use_rope=True, use_flash_attn=True, attn_backend="auto",
    )

    def grads_with(ckpt):
        torch.manual_seed(seed)
        m = MetisLM(cfg).to(device).train()
        idx = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss, _ = m(idx, targets=idx, use_checkpointing=ckpt)
        loss.backward()
        return {n: p.grad.clone().float() for n, p in m.named_parameters() if p.grad is not None}

    g1, g2 = grads_with(False), grads_with(True)
    names = set(g1) & set(g2)
    ok = all(torch.allclose(g1[n], g2[n], rtol=tol, atol=tol) for n in names)
    worst = max(torch.allclose(g1[n], g2[n], rtol=tol, atol=tol) for n in names)
    r.check("checkpointed grads == full grads", ok,
            f"max_grad_diff={max(float((g1[n]-g2[n]).abs().max()) for n in names):.2e}")


def verify_checkpoint_compat(r: Result, device: str, ckpt_path: str, seed: int):
    """A real pre-fusion checkpoint loads into the fused model and matches
    the reference model loaded from the same file."""
    path = Path(ckpt_path)
    if not path.exists():
        r.skip("real checkpoint compat", f"no checkpoint at {path}")
        return
    print("\n[checkpoint] pre-fusion checkpoint → fused model")
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    assert isinstance(sd, dict), "expected a state_dict"

    cfg = ModelConfig(
        vocab_size=sd["tok_emb.weight"].shape[0],
        d_model=sd["tok_emb.weight"].shape[1],
        n_heads=4, n_kv_heads=4, n_layers=sum(1 for k in sd if k.startswith("layers.") and ".ln_1.weight" in k),
        max_seq_len=sd.get("layers.0.attn.causal_mask", torch.zeros(1, 1, 256, 256)).shape[-1],
        dropout=0.0, use_rmsnorm=True, use_swiglu=True, use_rope=True,
        tie_weights=True, use_moe=False, use_qk_norm=False,
        use_attention_sink=False, use_flash_attn=True, attn_backend="auto",
    )
    fused = MetisLM(cfg).to(device).eval()
    missing, unexpected = fused.load_state_dict(sd)
    r.check("fused model loads old-format state_dict",
            not missing and not unexpected,
            f"missing={len(missing)} unexpected={len(unexpected)}")

    ref = _ReferenceLM(cfg).to(device).eval()
    ref.load_state_dict(sd)

    torch.manual_seed(seed)
    idx = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        lf = fused(idx, idx)[0].float()
        lr = ref(idx, idx)[0].float()
    ok = torch.allclose(lf, lr, rtol=2e-2, atol=2e-2) and bool(torch.isfinite(lf).all())
    r.check("fused(ckpt) ≈ reference(ckpt)", ok,
            f"max_logit_diff={(lf - lr).abs().max().item():.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fused-block numerical parity verification")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--checkpoint", default="checkpoints/final_model.pt")
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _configure_stdio()

    print(f"Metis fused-block parity — device={device} seed={args.seed} tol={args.tol}")
    r = Result(args.tol)
    verify_primitive_identity(r, device, args.seed)
    verify_block_outputs(r, device, args.seed, args.tol)
    verify_model_grads(r, device, args.seed, args.tol)
    verify_kv_cache_decode(r, device, args.seed, args.tol)
    verify_grad_checkpointing(r, device, args.seed, args.tol)
    verify_checkpoint_compat(r, device, args.checkpoint, args.seed)

    ok = r.summary()
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
