"""
Μῆτις (Metis) — Advanced Transformer Model
=============================================
A modern GPT-style decoder-only transformer with:
  • RMSNorm (pre-norm architecture)
  • Rotary Position Embeddings (RoPE)
  • SwiGLU feed-forward network
  • Grouped Query Attention (GQA) / MQA / MHA
  • KV-Cache for efficient inference
  • Gradient checkpointing
  • Weight tying
  • Mixture of Experts (MoE) — optional
  • QK-Normalization — optional, for training stability
  • Attention Sink — optional, for extended context
  • Flash Attention v2 via PyTorch SDPA
"""

import logging
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attn import causal_attention, set_backend_flags
from .kv import KVCache, LayerKV, cached_len_of  # KV cache subsystem (Phase 7)
from .moe import MoE  # grouped execution engine (token sorting / grouped GEMM)

# Chunk the training cross-entropy whenever the full (B*T, vocab) logits would
# exceed this (fp32) byte footprint, so a 100k-vocab model never materializes a
# multi-GB logits tensor on a 16 GB GPU. Small models stay on the original path
# (which also returns the full ``logits`` tensor).
_CE_CHUNK_MIN_BYTES = 256 * 1024 * 1024
# Cap each loss chunk's logits at ~32M elements (~64 MB fp16 / 128 MB fp32).
_CE_CHUNK_ELEMS = 32_000_000

logger = logging.getLogger("metis.model")

# ──────────────────────────────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Forward is a single fused kernel via ``F.rms_norm`` (torch ≥ 2.4) instead
    of the old multi-op formulation (``pow → mean → add → rsqrt → mul``), which
    launched ~7 elementwise/reduction kernels per norm. ``F.rms_norm`` computes
    the same value with fp32 accumulation and is bit-identical to the manual
    version for fp32/fp16/bf16 inputs — verified in ``benchmarks/verify_block_parity.py``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)


def precompute_rope_frequencies(
    dim: int, max_seq_len: int, theta: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Precompute complex-valued RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(
    x: torch.Tensor, freqs: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply rotary position embeddings to query/key tensors.

    ``position_ids`` (``(B, T)``) optionally overrides the default implicit
    0..T-1 positions — used for packed training where every document segment
    restarts its RoPE clock (see ``metis/packing.py``).
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if position_ids is None:
        freqs = freqs[: x.shape[2], :].unsqueeze(0).unsqueeze(0)
    else:
        freqs = freqs[position_ids].unsqueeze(1)  # (B, 1, T, D/2)
    x_rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return x_rotated.type_as(x)


def apply_rope_pair(
    q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to *q* and *k* jointly — 4 kernels instead of 6.

    The per-tensor path (``apply_rope``) does ``float() → complex-mul →
    type_as`` × 2 tensors = 6 kernel launches.  Concatenating along the head
    dim first lets the three data-dependent ops (cast-up, multiply, cast-down)
    each run on the combined buffer — saving 2 launches per block layer.

    ``split`` is pure metadata (zero kernels).  The function is bit-identical
    to two separate ``apply_rope`` calls.

    ``position_ids`` (``(B, T)``) optionally overrides the default implicit
    0..T-1 positions so RoPE restarts at every packed-document boundary.
    """
    if position_ids is None:
        freqs = freqs[: q.shape[2], :].unsqueeze(0).unsqueeze(0)
    else:
        freqs = freqs[position_ids].unsqueeze(1)  # (B, 1, T, D/2)
    qk = torch.cat([q, k], dim=1)                       # 1 kernel (memcpy)
    qk_c = torch.view_as_complex(                        # metadata
        qk.float().reshape(*qk.shape[:-1], -1, 2))       # 1 kernel (cast-up)
    out = torch.view_as_real(qk_c * freqs).flatten(-2)    # 1 kernel (cmul)
    out = out.type_as(qk)                                # 1 kernel (cast-down)
    q_rot, k_rot = out.split([q.shape[1], k.shape[1]], dim=1)  # metadata
    return q_rot, k_rot


# ──────────────────────────────────────────────────────────────────────────────
# QK-Normalization (Phase 3)
# ──────────────────────────────────────────────────────────────────────────────

class QKNorm(nn.Module):
    """Query/Key normalization for attention training stability.

    Normalizes the query and key projections before RoPE to prevent
    attention logits from growing too large. Used in LLaMA 3+.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm over the last dimension (head_dim).

        Args:
            x: Tensor of shape (B, n_heads, T, head_dim)
        Returns:
            Normalized tensor, same shape, *in the input dtype*.

        ``RMSNorm`` normalizes in fp32 but its final ``* weight`` step promotes
        an fp16/bf16 input to fp32 under AMP autocast (the fp32 weight widens
        the elementwise multiply). Casting back to the input dtype here keeps
        q/k in the projected dtype so they stay eligible for the fused
        fp16/bf16 attention kernels — without this cast, ``use_qk_norm``
        silently forced every attention call down to the fp32 manual path.
        """
        return self.norm(x).type_as(x)


# ──────────────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with GQA, RoPE, KV-cache, QK-Norm.

    Supports:
      • MHA  (n_kv_heads == n_heads)
      • GQA  (n_kv_heads <  n_heads)
      • MQA  (n_kv_heads == 1)
      • QK-Normalization (optional, via config.use_qk_norm)
      • Flash Attention via PyTorch SDPA
    """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0

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
        self.last_backend = None  # concrete kernel used by the last forward

        # Fused QKV projection — a single GEMM instead of three (q_proj +
        # k_proj + v_proj). The state_dict contract still uses the legacy split
        # keys (see the compat shim below), so this is invisible to checkpoints.
        self.qkv = nn.Linear(
            config.d_model, config.d_model + 2 * self.kv_dim, bias=False
        )
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self._register_state_dict_hook(CausalSelfAttention._qkv_rename_hook)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # QK-Normalization
        if self.use_qk_norm:
            self.q_norm = QKNorm(self.head_dim)
            self.k_norm = QKNorm(self.head_dim)

        # Causal mask (legacy manual path). Kept registered — it is a
        # persistent buffer and therefore part of every existing checkpoint's
        # state_dict; removing it would break strict loading of old checkpoints.
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len),
        )

        # RoPE frequencies (extra slot for attention sink — adds 1 token)
        if self.use_rope:
            rope_len = config.max_seq_len + (1 if self.use_attention_sink else 0)
            self.register_buffer(
                "rope_freqs",
                precompute_rope_frequencies(self.head_dim, rope_len),
                persistent=False,
            )

        # Attention sink token (first token in sequence)
        self.use_attention_sink = config.use_attention_sink
        if self.use_attention_sink:
            # Learnable sink token — appended to every sequence
            self.sink_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

    # ── Checkpoint compatibility (fused QKV ↔ legacy split q/k/v) ────────────
    #
    # Internally Q/K/V share one fused ``self.qkv`` Linear (fewer kernels), but
    # the public state_dict contract keeps the OLD split keys
    # (``q_proj.weight`` / ``k_proj.weight`` / ``v_proj.weight``). Every
    # pre-fusion checkpoint therefore loads unchanged, and checkpoints written
    # now stay readable by pre-fusion code. Loading accepts either format.
    #
    # Save side: a state_dict hook renames ``qkv.weight`` → the three split
    # keys (must be a hook, not ``_save_to_state_dict`` — the ``qkv`` child
    # module's own recursion would re-emit ``qkv.weight`` afterwards).
    #
    # Load side: ``_load_from_state_dict`` translates the split keys back into
    # a single ``qkv.weight`` in the (mutable) shared state dict before the
    # standard recursion consumes it.

    @staticmethod
    def _qkv_rename_hook(module, state_dict, prefix, local_metadata):
        key = prefix + "qkv.weight"
        if key in state_dict:
            # Weight shape is (out_features, in_features) = (d + 2·kv, D).
            # Split along the output-dim (rows) to recover the legacy keys.
            fused = state_dict.pop(key)
            d, kv = module.d_model, module.kv_dim
            state_dict[prefix + "q_proj.weight"] = fused[:d]
            state_dict[prefix + "k_proj.weight"] = fused[d : d + kv]
            state_dict[prefix + "v_proj.weight"] = fused[d + kv :]
        return state_dict

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        qkv_key = prefix + "qkv.weight"
        split_keys = [
            prefix + n + ".weight" for n in ("q_proj", "k_proj", "v_proj")
        ]
        if all(k in state_dict for k in split_keys):
            # Legacy checkpoint: fuse the three projections into one.
            # Each weight is (out_features, in_features); concat on the
            # output-dim (rows) to match the fused Linear's layout.
            state_dict[qkv_key] = torch.cat(
                [state_dict.pop(k) for k in split_keys], dim=0
            )
        elif qkv_key not in state_dict:
            # Neither format fully present — let strict reporting surface it.
            missing_keys.extend(k for k in split_keys if k not in state_dict)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | LayerKV | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | LayerKV | None]:
        B, T, C = x.size()

        if attention_mask is not None and self.use_attention_sink:
            raise RuntimeError(
                "Packed attention masks are incompatible with the attention "
                "sink (the sink prepends a token outside the packed layout). "
                "Disable use_attention_sink or use_packing."
            )

        # KV-cache backend selection (Phase 7): a ``LayerKV`` is the static or
        # quantized cache (preallocated buffers / int8 compression, see
        # ``metis/kv.py``); a ``(K, V)`` tuple is the legacy growable cache.
        # ``cached_len`` drives both RoPE offsets and the attention-sink
        # "cold start" decision (an empty LayerKV is a cold start, just like
        # ``kv_cache=None``).
        is_layer_cache = isinstance(kv_cache, LayerKV)
        is_cold = kv_cache is None or (is_layer_cache and kv_cache.cached_len == 0)

        # Attention sink: prepend a learnable sink token on cold starts
        if self.use_attention_sink and is_cold:
            sink = self.sink_token.expand(B, -1, -1)
            x = torch.cat([sink, x], dim=1)
            T = x.size(1)

        # Project Q, K, V — one fused GEMM, then split into the three views.
        q, k, v = self.qkv(x).split([self.d_model, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Normalization (before RoPE)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE — fused pair path: 4 kernels instead of 6.
        # The rotation is computed in fp32 internally (complex math) and cast
        # back to the *projected* dtype (fp16/bf16 under AMP) — never to the
        # layer-input dtype.
        if self.use_rope:
            if position_ids is not None:
                # Packed training: positions reset at every document boundary,
                # so each segment's tokens rotate as if they were a standalone
                # sequence starting at position 0.
                q, k = apply_rope_pair(q, k, self.rope_freqs, position_ids=position_ids)
            elif not is_cold:
                offset = kv_cache.cached_len if is_layer_cache else kv_cache[0].size(2)
                if offset + T > self.rope_freqs.size(0):
                    raise RuntimeError(
                        f"RoPE position {offset + T} exceeds precomputed buffer "
                        f"({self.rope_freqs.size(0)}). Generation grew past max_seq_len."
                    )
                rope_slice = self.rope_freqs[offset: offset + T, :]
                q, k = apply_rope_pair(q, k, rope_slice)
            else:
                q, k = apply_rope_pair(q, k, self.rope_freqs)

        # KV-Cache — always returned so a first (cold) forward hands a populated
        # cache to the caller, enabling incremental decode. Stored with the
        # unexpanded n_kv_heads (GQA expansion happens inside the attention
        # dispatcher only when a backend needs it).
        if is_layer_cache:
            # Static / quantized: write in place, read back the live prefix.
            kv_cache.append(k, v)
            k, v = kv_cache.keys_values()
            new_kv_cache = kv_cache
            attn_mask = None  # decode/prefix semantics (no packed mask)
        elif kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
            new_kv_cache = (k, v)
            attn_mask = None
        else:
            new_kv_cache = (k, v)
            attn_mask = attention_mask

        # Attention computation — dispatched: flash-attn → SDPA → math.
        # The manual masked-softmax path (math) is bit-identical to the legacy
        # implementation; fused kernels differ only in floating-point rounding.
        # For packed training an explicit block-diagonal causal mask replaces
        # ``is_causal``; it is never applied during decode (kv_cache).
        dropout_p = self.attn_dropout.p if self.training else 0.0
        backend_log = []
        y = causal_attention(
            q, k, v,
            dropout_p=dropout_p,
            is_causal=True,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            backend=self.backend_request,
            use_flash_attn=self.use_flash_attn,
            training=self.training,
            out_backend=backend_log,
            attention_mask=attn_mask,
        )
        self.last_backend = backend_log[0] if backend_log else None

        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        y = self.resid_dropout(self.o_proj(y))

        # Remove attention sink token from output (it was prepended on cold
        # starts — training, or an empty cache under a static/quantized backend).
        if self.use_attention_sink and is_cold:
            y = y[:, 1:, :]
        # Warm decode: sink stays in KV-cache but shouldn't expand output

        return y, new_kv_cache


# ──────────────────────────────────────────────────────────────────────────────
# Feed-Forward Networks
# ──────────────────────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer, 2020).

    The gate (``w1``) and up (``w3``) projections share one fused ``w13``
    Linear — a single GEMM instead of two. The state_dict contract keeps the
    legacy split keys (see the compat shim below), so checkpoints are
    unaffected. ``w2`` stays a separate projection (it receives the scaled
    init in ``MetisLM._init_weights``).
    """

    def __init__(self, config):
        super().__init__()
        self.hidden = int(4 * config.d_model * 2 / 3)
        self.hidden = ((self.hidden + 7) // 8) * 8
        hidden = self.hidden

        self.w13 = nn.Linear(config.d_model, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self._register_state_dict_hook(SwiGLU._w13_rename_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).split(self.hidden, dim=-1)
        return self.dropout(self.w2(F.silu(gate) * up))

    # ── Checkpoint compatibility (fused gate/up ↔ legacy w1/w3) ──────────────
    # Mirrors ``CausalSelfAttention``: emit the old ``w1.weight`` /
    # ``w3.weight`` keys on save and accept either format on load.

    @staticmethod
    def _w13_rename_hook(module, state_dict, prefix, local_metadata):
        key = prefix + "w13.weight"
        if key in state_dict:
            # Weight shape is (out_features, in_features) = (2·hidden, D).
            # Split along the output-dim (rows) to recover gate / up weights.
            fused = state_dict.pop(key)
            h = module.hidden
            state_dict[prefix + "w1.weight"] = fused[:h]
            state_dict[prefix + "w3.weight"] = fused[h:]
        return state_dict

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        w13_key = prefix + "w13.weight"
        split_keys = [prefix + n + ".weight" for n in ("w1", "w3")]
        if all(k in state_dict for k in split_keys):
            # Each weight is (out_features, in_features); concat on the
            # output-dim (rows) to form the fused weight.
            state_dict[w13_key] = torch.cat(
                [state_dict.pop(k) for k in split_keys], dim=0
            )
        elif w13_key not in state_dict:
            missing_keys.extend(k for k in split_keys if k not in state_dict)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


class MLP(nn.Module):
    """Standard GELU feed-forward network (fallback)."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


# ──────────────────────────────────────────────────────────────────────────────
# Mixture of Experts (MoE) — Phase 3
# ──────────────────────────────────────────────────────────────────────────────
#
# The ``MoE`` module now lives in ``metis/moe.py`` with a grouped execution
# engine (token sorting → expert batching → grouped GEMM → grouped SwiGLU →
# grouped output projection) replacing the old per-expert loop. ``per_expert``
# is retained there as the byte-identical reference path. It is imported at
# the top of this module; ``TransformerBlock`` below constructs it directly.


# ──────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ──────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with residual connections."""

    def __init__(self, config):
        super().__init__()
        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.ln_1 = norm_cls(config.d_model)
        # Multi-head Latent Attention (kv_backend="mla") is an architecture
        # swap: same API, different parameters (see metis/mla.py). Imported
        # lazily to avoid the mla -> model import cycle.
        if config.kv_backend == "mla":
            from .mla import MLAAttention
            self.attn = MLAAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = norm_cls(config.d_model)
        self.ffn = MoE(config) if config.use_moe else \
                   SwiGLU(config) if config.use_swiglu else MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, object]:
        attn_out, new_cache = self.attn(
            self.ln_1(x), kv_cache=kv_cache,
            attention_mask=attention_mask, position_ids=position_ids,
        )
        x = x + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x, new_cache


# ──────────────────────────────────────────────────────────────────────────────
# Metis Language Model
# ──────────────────────────────────────────────────────────────────────────────

class MetisLM(nn.Module):
    """
    Μῆτις v3.0 — Advanced decoder-only transformer language model.

    Architecture:
      • Pre-norm with RMSNorm
      • RoPE position embeddings
      • SwiGLU or MoE feed-forward
      • GQA / MQA / MHA attention
      • QK-Normalization (optional)
      • Attention sink (optional)
      • Weight tying
      • Flash Attention via SDPA
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Layer prefetching: while layer N computes, speculatively warm layer
        # N+1's expert cache on a side stream (CUDA only; on CPU it is a
        # synchronous warm-up). Only meaningful with MoE + the expert cache.
        self._layer_prefetch = None
        if getattr(config, "use_layer_prefetch", True):
            _force_cpu = os.environ.get("METIS_LAYER_PREFETCH_FORCE_CPU", "0") == "1"
            if torch.cuda.is_available() or _force_cpu:
                from .layer_prefetch import LayerExpertPrefetcher

                self._layer_prefetch = LayerExpertPrefetcher(self.layers)
                for i, layer in enumerate(self.layers):
                    ffn = getattr(layer, "ffn", None)
                    if hasattr(ffn, "experts"):
                        ffn._prefetcher = self._layer_prefetch
                        ffn._prefetch_idx = i

        # Final norm
        norm_cls = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.norm_f = norm_cls(config.d_model)

        # Output head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        if config.tie_weights:
            self.tok_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if (
                pn.endswith("o_proj.weight")
                or pn.endswith("w2.weight")
                or pn.endswith("c_proj.weight")
            ):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        # Compute parameter count. With tied embeddings tok_emb.weight and
        # lm_head.weight are the SAME tensor, and parameters() dedups shared
        # tensors, so this sum already counts the embedding exactly once.
        config.n_params = self._format_params(
            sum(p.numel() for p in self.parameters()))

        # Configure the global SDPA kernel flags to match the requested
        # attention backend (see metis/attn.py). No-op on CPU / old torch.
        if hasattr(F, "scaled_dot_product_attention"):
            set_backend_flags(getattr(config, "attn_backend", "auto"))

    @staticmethod
    def _format_params(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.1f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_checkpointing: bool = False,
        kv_cache: list | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list | None]:
        """
        Forward pass.

        Args:
            idx: Input token indices (B, T).
            targets: Target token indices (B, T) for loss computation. Padding
                positions (``<pad>``, id 0) are ignored by the loss.
            use_checkpointing: Enable gradient checkpointing.
            kv_cache: List of (K, V) tuples per layer for cached inference.
            attention_mask: Optional ``(B, 1, T, T)`` bool block-diagonal causal
                mask for packed training (see ``metis/packing.py``). When
                ``None`` the standard causal mask applies.
            position_ids: Optional ``(B, T)`` long RoPE positions for packed
                training — reset at every document boundary so each segment
                starts at position 0.

        Returns:
            Tuple of (logits, loss, new_kv_cache). With ``targets`` given, a
            large model computes the loss via a memory-efficient chunked
            cross-entropy and ``logits`` is ``None`` (the full (B, T, V) tensor
            is never built); for small models ``logits`` is the full tensor.
            With ``targets=None`` ``logits`` covers just the final position.
        """
        B, T = idx.size()
        device = idx.device

        if T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
            )

        if attention_mask is not None and attention_mask.size(-1) != T:
            raise ValueError(
                f"attention_mask sequence dim {attention_mask.size(-1)} does "
                f"not match input length {T}"
            )

        # Embeddings. With RoPE disabled, the position embedding must reflect
        # the token's ABSOLUTE position — during warm decode that is the cached
        # prefix length (``cached_len + i``), not a fresh 0..T-1 each call
        # (which would embed every generated token at position 0 and corrupt
        # long generations). ``cached_len_of`` returns 0 for cold starts and
        # training, so the common paths are unchanged.
        x = self.tok_emb(idx)
        if not self.config.use_rope:
            if position_ids is not None:
                pos = position_ids
            else:
                base = cached_len_of(kv_cache)
                pos = base + torch.arange(0, T, dtype=torch.long, device=device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        # Transformer layers. The KV cache is always built and returned so the
        # first call (kv_cache=None) hands a populated cache back to the caller
        # (generate_text relies on this to start decoding incrementally).
        #
        # Cache container per backend (Phase 7):
        #   "default" / "mla" — a plain list of per-layer caches, rebuilt each
        #                       forward (byte-identical legacy behavior).
        #   "static" / "quantized" — a shared metis.kv.KVCache object (list-like)
        #                       whose per-layer buffers are written in place.
        backend = self.config.kv_backend
        if backend in ("default", "mla"):
            container = kv_cache  # None → cold start, or a legacy list
        elif kv_cache is None:
            container = None if (self.training or targets is not None) else \
                KVCache(backend, self.config, len(self.layers))
        elif isinstance(kv_cache, KVCache):
            container = kv_cache
        else:
            # Warm start from a legacy [(K, V), ...] list (e.g. a chunked
            # prefill or a scheduler-produced list) — re-appended on ingestion,
            # so a legacy cache is compressed in place.
            container = KVCache.from_legacy(
                backend, self.config, kv_cache, len(self.layers))

        new_kv_cache = []
        for i, layer in enumerate(self.layers):
            # Layer prefetch: warm layer i+1's expert cache on a side stream
            # while layer i computes (no-op without a prefetcher / CUDA graphs).
            if self._layer_prefetch is not None:
                self._layer_prefetch.prefetch_next(i)
            layer_cache = container[i] if container is not None else None
            if use_checkpointing and self.training:
                x, _ = checkpoint(
                    layer, x, layer_cache, attention_mask, position_ids,
                    use_reentrant=False,
                )
            else:
                x, new_cache = layer(
                    x, kv_cache=layer_cache,
                    attention_mask=attention_mask, position_ids=position_ids,
                )
                if backend in ("default", "mla"):
                    new_kv_cache.append(new_cache)

        x = self.norm_f(x)

        if targets is not None:
            V = self.lm_head.weight.size(0)
            # Memory-efficient cross-entropy: materializing the full (B*T, vocab)
            # logits for a 100k-vocab model is ~0.8 GB in fp16 (autocast then casts
            # them to fp32 for the softmax → ~1.6 GB), and the softmax's fp32
            # temporaries multiply that several-fold — on top of an 805M model
            # (fp32 weights + grads + bnb 8-bit states ≈ 8 GB) that OOMs a 16 GB
            # GPU on the first training step. When the full logits would be large
            # (≥ _CE_CHUNK_MIN_BYTES fp32), apply lm_head + CE in chunks over T:
            # identical loss (sum over non-padded tokens / their count) with a
            # one-chunk peak. Small models keep the old path so ``logits`` is
            # still returned with targets.
            full_bytes = B * T * V * 4
            if full_bytes >= _CE_CHUNK_MIN_BYTES:
                chunk_T = max(1, min(T, _CE_CHUNK_ELEMS // max(1, B * V)))
                loss_sum = torch.zeros((), device=x.device)  # fp32 accumulator
                n_valid = 0
                for s in range(0, T, chunk_T):
                    e = min(s + chunk_T, T)
                    tgt = targets[:, s:e].reshape(-1)
                    chunk_logits = self.lm_head(x[:, s:e]).reshape(-1, V)
                    loss_sum = loss_sum + F.cross_entropy(
                        chunk_logits, tgt,
                        ignore_index=self.config.pad_id, reduction="sum",
                    )
                    n_valid += (tgt != self.config.pad_id).sum()
                loss = loss_sum / n_valid.clamp(min=1)
                # The full (B, T, V) logits are not materialized here — training
                # /val callers only consume ``loss`` anyway.
                logits = None
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, V),
                    targets.view(-1),
                    ignore_index=self.config.pad_id,
                )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if backend in ("default", "mla"):
            return logits, loss, new_kv_cache
        return logits, loss, container

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device_type: str,
        optimizer: str = "adamw",
    ) -> torch.optim.Optimizer:
        """Configure an AdamW-family optimizer with weight decay separation.

        ``optimizer="bnb8bit"`` uses bitsandbytes' 8-bit AdamW, which stores the
        optimizer states in int8 — required to fit ~1B-param models in 16 GB of
        VRAM. It needs CUDA and bitsandbytes; either missing → falls back to
        plain AdamW with a warning.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if optimizer == "bnb8bit":
            if not device_type.startswith("cuda"):
                logger.warning("bnb8bit requires CUDA — falling back to AdamW")
                optimizer = "adamw"
            else:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    logger.warning(
                        "bitsandbytes is not installed — falling back to AdamW. "
                        "Install it (`pip install bitsandbytes`) to fit ~1B-param "
                        "models in 16 GB of VRAM."
                    )
                    optimizer = "adamw"
                else:
                    return bnb.optim.AdamW8bit(
                        optim_groups,
                        lr=learning_rate,
                        betas=(0.9, 0.95),
                        eps=1e-8,
                    )

        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type.startswith("cuda")

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )
        return optimizer

    def count_parameters(self) -> dict:
        """Return detailed parameter counts by component."""
        counts = {}
        for name, param in self.named_parameters():
            component = name.split(".")[0]
            counts[component] = counts.get(component, 0) + param.numel()
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def invalidate_moe_caches(self) -> None:
        """Invalidate all MoE expert weight caches across all layers.

        Call after each ``optimizer.step()`` / ``scaler.step()`` when using a
        fused CUDA optimiser, which mutates weights without bumping their
        ``_version`` counter.  The framework's own ``training.py`` loop does
        this automatically; custom training loops should call it after each
        weight update.
        """
        for layer in self.layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "invalidate_cache"):
                ffn.invalidate_cache()

    def get_moe_cache_stats(self) -> list[dict | None]:
        """Return a list of cache stats dicts, one per MoE layer.

        ``None`` entries correspond to non-MoE or cache-disabled layers.
        """
        stats = []
        for layer in self.layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "cache_stats"):
                stats.append(ffn.cache_stats())
            else:
                stats.append(None)
        return stats

    def get_attention_backend(self) -> dict:
        """Report the attention backend configuration and last-used kernel.

        Returns a dict combining the machine capability report
        (:func:`metis.attn.detect_attention_backends`) with the requested
        backend from config and the concrete kernel most recently used by any
        layer (``None`` before the first forward).
        """
        from .attn import detect_attention_backends

        info = detect_attention_backends()
        info["requested"] = getattr(self.config, "attn_backend", "auto")
        last_used = None
        for layer in self.layers:
            used = getattr(getattr(layer, "attn", None), "last_backend", None)
            if used:
                last_used = used
        info["last_used"] = last_used
        return info


# Backward compatibility
TinyLLM = MetisLM
