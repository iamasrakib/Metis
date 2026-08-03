"""
Μῆτις (Metis) — Multi-head Latent Attention (MLA)
====================================================
A from-scratch implementation of the DeepSeek-V2/V3 attention mechanism
[arXiv:2405.04434], simplified for a tiny model: no query-side latent
(``c_t^Q``), a single shared latent per layer, and per-head key/value
up-projections.

Why
---
Standard MHA stores **two** full tensors per token per layer in the KV cache
(``2 * n_heads * head_dim`` values). GQA shrinks this by sharing KV heads
(``2 * n_kv_heads * head_dim``) at the cost of reduced expressiveness. MLA
instead *compresses* the KV state into a low-rank latent vector shared across
all heads and caches only that latent plus the RoPE part of the key:

    c_t     = W_DKV h_t                                  # (c_d,) shared latent
    k_t^C   = W_UK c_t         (content part,  not rotated)
    k_t^R   = RoPE(W_KR c_t)   (rope part,     rotated)
    v_t     = W_UV c_t

Per token per layer the cache holds ``c_d + n_heads * rope_head_dim`` values
instead of ``2 * n_heads * head_dim``. Because the latent is *shared across
heads* while GQA shares the *whole key/value*, MLA keeps per-head
expressiveness at a fraction of the cache — the win widens as ``n_heads``
grows. At the small presets it is comparable to or smaller than GQA (see
``docs/mla.md`` for the ratio table).

Weight absorption
-----------------
At decode time the content key is never materialized from the latent. The
query is projected into latent space once per layer and attends against the
cached latent directly:

    q_latent = W_UKᵀ q_content            (folded into the query projection)
    score    = q_latent · c  +  RoPE(q_R) · k^R

and the value up-projection is folded into the output projection
(``W_OV = W_O · W_UV``) so attention output is produced straight from the
latent:

    o_lat = softmax(score) · c
    u     = W_OV · o_lat

This is mathematically identical to reconstructing k/v and running ordinary
attention — verified bit-tight in ``benchmarks/verify_kv_parity.py``.

API / training notes
--------------------
MLA is an **architecture** change (new per-layer parameters), not a cache-only
optimisation: a model built with ``kv_backend="mla"`` has different weights
from a GQA/MHA checkpoint and must be **trained from scratch**. The public
interface is unchanged — ``MetisLM.forward(..., kv_cache=...) -> (logits,
loss, new_kv_cache)`` with a per-layer cache object — so ``generate_text``,
the server and the scheduler work without modification.

Prefill uses the explicit K/V path and dispatches through
:func:`metis.attn.causal_attention` (so fused FlashAttention kernels engage on
CUDA); decode uses the absorbed path with only the latent cache.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attn import causal_attention
from .model import apply_rope, precompute_rope_frequencies

__all__ = ["MLAAttention", "MLALayerCache"]


class MLALayerCache:
    """One layer's MLA cache: ``(latent c, rope keys k^R)``.

    Index-compatible with a 2-tuple and exposes ``cached_len`` so generic
    cache consumers (e.g. ``metis.kv.cached_len_of``) can read the live
    context length without knowing the layout.
    """

    __slots__ = ("c", "k_rope")

    def __init__(self, c: torch.Tensor, k_rope: torch.Tensor):
        self.c = c            # (B, T, c_d) — compressed latent, shared across heads
        self.k_rope = k_rope  # (B, n_heads, T, rope_head_dim) — rotated rope keys

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int) -> torch.Tensor:
        if not 0 <= i <= 1:
            raise IndexError(f"MLALayerCache index {i} out of range")
        return (self.c, self.k_rope)[i]

    @property
    def cached_len(self) -> int:
        return self.k_rope.size(2)

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (f"MLALayerCache(len={self.cached_len}, "
                f"c={tuple(self.c.shape)}, k_rope={tuple(self.k_rope.shape)})")


class MLAAttention(nn.Module):
    """Multi-head latent attention with compressed KV cache.

    Args come from ``ModelConfig``. ``mla_kv_latent_dim`` (default
    ``d_model // n_heads``) is the shared latent dimension ``c_d``;
    ``mla_rope_head_dim`` (default ``head_dim // 2``, must be even) is the
    RoPE part of the key head dim.
    """

    def __init__(self, config):
        super().__init__()
        if not config.use_rope:
            raise ValueError("MLAAttention requires use_rope=True (RoPE is "
                             "essential to the rope-split key decomposition).")
        if config.n_kv_heads not in (0, config.n_heads):
            # MLA shares one latent across all heads; a separate KV-head count
            # has no meaning here. Requiring n_heads keeps the module honest.
            raise ValueError(
                "MLAAttention uses a shared latent across all query heads; "
                f"set n_kv_heads={config.n_heads} (MHA) or leave it at the "
                f"default (got {config.n_kv_heads})."
            )

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads          # content dim (and v dim)
        self.rope_head_dim = config.mla_rope_head_dim or (self.head_dim // 2)
        if self.rope_head_dim <= 0 or self.rope_head_dim % 2 != 0:
            raise ValueError(f"mla_rope_head_dim must be > 0 and even, got "
                             f"{self.rope_head_dim}")
        self.c_d = config.mla_kv_latent_dim or (config.d_model // config.n_heads)
        if self.c_d <= 0:
            raise ValueError(f"mla_kv_latent_dim must be > 0, got {self.c_d}")
        self.q_head_dim = self.head_dim + self.rope_head_dim      # total query head dim
        self.use_attention_sink = config.use_attention_sink
        self.scale = 1.0 / math.sqrt(
            self.head_dim + self.rope_head_dim if config.mla_scale_head_dim
            else self.head_dim
        )
        self.backend_request = getattr(config, "attn_backend", "auto")
        self.use_flash_attn = config.use_flash_attn
        self.last_backend = None   # concrete kernel used by the last forward

        # Projections (all bias-free, like the rest of Metis).
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.q_head_dim, bias=False)
        self.kv_latent_proj = nn.Linear(config.d_model, self.c_d, bias=False)   # W_DKV
        self.k_content_proj = nn.Linear(self.c_d, self.n_heads * self.head_dim, bias=False)  # W_UK
        self.k_rope_proj = nn.Linear(  # W_KR: latent -> per-head rope key part
            self.c_d, self.n_heads * self.rope_head_dim, bias=False)
        self.v_proj = nn.Linear(self.c_d, self.n_heads * self.head_dim, bias=False)  # W_UV
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE frequencies over the rope part only (+1 slot for the sink).
        rope_len = config.max_seq_len + (1 if self.use_attention_sink else 0)
        self.register_buffer(
            "rope_freqs",
            precompute_rope_frequencies(self.rope_head_dim, rope_len),
            persistent=False,
        )

        # Attention sink token (same contract as CausalSelfAttention).
        if self.use_attention_sink:
            self.sink_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Absorbed W_OV = W_O · W_UV, cached across decodes and recomputed
        # lazily whenever the weights change (version guard — a stale fold is
        # never used, even after in-place mutations).
        self._cached_w_ov = None
        self._cached_w_ov_version = -1

    # ── helpers ───────────────────────────────────────────────────────────

    def _weight_views(self):
        """Per-head views of the up-projection and output weights."""
        uk = self.k_content_proj.weight.view(self.n_heads, self.head_dim, self.c_d)
        uv = self.v_proj.weight.view(self.n_heads, self.head_dim, self.c_d)
        o = self.o_proj.weight.view(self.d_model, self.n_heads, self.head_dim)
        return uk, uv, o

    def _w_ov(self) -> torch.Tensor:
        """Folded output projection ``W_OV[d,h,c] = Σ_k W_O[d,h,k] W_UV[h,k,c]``.

        ``(d_model, n_heads, c_d)``.
        """
        version = self.o_proj.weight._version + self.v_proj.weight._version
        if self._cached_w_ov is None or self._cached_w_ov_version != version:
            _, uv, o = self._weight_views()
            self._cached_w_ov = torch.einsum("dhk,hkc->dhc", o, uv)
            self._cached_w_ov_version = version
        return self._cached_w_ov

    def _rotate(self, t: torch.Tensor, offset: int | None = None,
                position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Apply RoPE to the rope part ``t`` (``(B, H, T, rope_head_dim)``)."""
        if position_ids is not None:
            return apply_rope(t, self.rope_freqs, position_ids=position_ids)
        if offset is not None:
            if offset + t.size(2) > self.rope_freqs.size(0):
                raise RuntimeError(
                    f"RoPE position {offset + t.size(2)} exceeds the precomputed "
                    f"buffer ({self.rope_freqs.size(0)}). Generation grew past "
                    f"max_seq_len."
                )
            return apply_rope(t, self.rope_freqs[offset: offset + t.size(2)])
        return apply_rope(t, self.rope_freqs)

    def _split_q(self, q):
        """Split the fused query into content / rope parts per head."""
        B, T, _ = q.shape
        q = q.view(B, T, self.n_heads, self.q_head_dim).transpose(1, 2)
        return q[..., : self.head_dim], q[..., self.head_dim:]  # (B,H,T,hD), (B,H,T,rD)

    def _mla_attention(self, scores, values, is_causal):
        """softmax → dropout → weighted sum over ``values``.

        ``values`` is the *latent* ``c`` for the absorbed decode path and the
        reconstructed ``v`` for prefill. A causal mask is applied only when
        ``T_q == T_k`` (prefill); decode attends to the full prefix.
        """
        T_q, T_k = scores.size(-2), scores.size(-1)
        if is_causal and T_q == T_k:
            mask = torch.tril(
                torch.ones(T_q, T_k, device=scores.device)).view(1, 1, T_q, T_k)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(scores, dim=-1)
        att = F.dropout(att, p=self.attn_dropout.p, training=self.training)
        return att @ values

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: MLALayerCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, MLALayerCache]:
        """Attention forward with an MLA (latent + rope-key) cache.

        ``kv_cache=None`` → cold prefill: explicit K/V, dispatched through
        :func:`causal_attention` (fused kernels eligible on CUDA), and a fresh
        compressed cache returned. ``kv_cache=MLALayerCache`` → warm decode:
        absorbed-path attention against the latent only.
        """
        B, T, _ = x.size()

        if attention_mask is not None and self.use_attention_sink:
            raise RuntimeError(
                "Packed attention masks are incompatible with the attention "
                "sink (see CausalSelfAttention)."
            )
        is_cold = kv_cache is None

        # Attention sink: prepend a learnable sink token on cold starts.
        if self.use_attention_sink and is_cold:
            x = torch.cat([self.sink_token.expand(B, -1, -1), x], dim=1)
            T = x.size(1)

        # Latent compression + fused query projection.
        c = self.kv_latent_proj(x)                       # (B, T, c_d)
        q = self.q_proj(x)                               # (B, T, n_heads * q_head_dim)
        q_content, q_rope = self._split_q(q)

        if is_cold:
            # ── Prefill: explicit keys/values, dispatched attention ──────
            k_content = self.k_content_proj(c).view(
                B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k_rope = self._rotate(
                self.k_rope_proj(c).view(
                    B, T, self.n_heads, self.rope_head_dim).transpose(1, 2),
                position_ids=position_ids,
            )
            v = self.v_proj(c).view(
                B, T, self.n_heads, self.head_dim).transpose(1, 2)
            q_rope = self._rotate(q_rope, position_ids=position_ids)

            k = torch.cat([k_content, k_rope], dim=-1)   # (B, H, T, hD + rD)
            q_full = torch.cat([q_content, q_rope], dim=-1)

            backend_log = []
            y = causal_attention(
                q_full, k, v,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
                n_heads=self.n_heads, n_kv_heads=self.n_heads,
                backend=self.backend_request, use_flash_attn=self.use_flash_attn,
                training=self.training,
                out_backend=backend_log,
                attention_mask=attention_mask,
            )
            self.last_backend = backend_log[0] if backend_log else None
            y = y.transpose(1, 2).contiguous().view(B, -1, self.d_model)
            y = self.resid_dropout(self.o_proj(y))
            new_cache = MLALayerCache(c, k_rope)
        else:
            # ── Decode: absorbed path against the latent cache ───────────
            offset = kv_cache.cached_len
            q_rope = self._rotate(q_rope, offset=offset)
            k_rope_new = self._rotate(
                self.k_rope_proj(c).view(
                    B, T, self.n_heads, self.rope_head_dim).transpose(1, 2),
                offset=offset,
            )
            c_all = torch.cat([kv_cache.c, c], dim=1)
            k_rope_all = torch.cat([kv_cache.k_rope, k_rope_new], dim=2)
            self.last_backend = "mla_absorbed"

            # Fold W_UK into the query: content scores against the latent.
            uk, _, _ = self._weight_views()
            q_latent = torch.einsum("bhqd,hdc->bhqc", q_content, uk)
            scores = (
                q_latent @ c_all.transpose(-2, -1)
                + q_rope @ k_rope_all.transpose(-2, -1)
            ) * self.scale

            # softmax over (latent + rope keys), weighted sum over the latent,
            # then the folded OV projection.
            o_lat = self._mla_attention(scores, c_all, is_causal=False)
            y = torch.einsum("dhc,bhqc->bqd", self._w_ov(), o_lat)

            new_cache = MLALayerCache(c_all, k_rope_all)

        if self.use_attention_sink and is_cold:
            y = y[:, 1:, :]      # sink was prepended to the input
        return y, new_cache
