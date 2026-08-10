"""
Μῆτις (Metis) — FlashAttention dispatch layer
===============================================
Selects the fastest available causal-attention backend at runtime, per machine
and per call (based on tensor dtype / shape / device).

Dispatch priority (best → worst):

  1. ``flash_attn``   — the dao-AILab FlashAttention-2 package. Optional and
                        Linux-only; it is not shipped in Windows wheels and is
                        deliberately left out of ``requirements.txt``.
  2. PyTorch SDPA fused kernels, via ``F.scaled_dot_product_attention``:
       - SDPA FLASH_ATTENTION — a FlashAttention-2 kernel (torch builds that
         compile it; e.g. the Linux CUDA wheels). Not present in the Windows
         2.6.x wheels.
       - SDPA EFFICIENT_ATTENTION — the fused memory-efficient kernel
         (FlashAttention-family online-softmax algorithm). Available on Ampere+
         CUDA builds, including the Windows wheels.
  3. the exact manual math implementation (the legacy reference path).

Fallback is automatic: an unavailable or ineligible kernel is skipped and the
computation degrades to the next best backend. Eligibility includes the call's
dtype — on Turing GPUs the fused kernels are fp16-only, so a bf16 call there
degrades to torch's SDPA math kernel (``SDPA_MATH``, any dtype) rather than
crashing with "No available kernel". ``use_flash_attn=False`` (or
``attn_backend="math"``) pins the byte-identical manual reference, which is
also the deterministic path for debugging and reproduction.

The module is a pure dispatch layer: it never touches RoPE, QK-norm, dropout
modules, or position bookkeeping — those stay in ``metis/model.py``. Tensors
arrive here with RoPE and QK-norm already applied.

Backend names (user-facing → concrete):
    "auto"           → best available: flash_attn pkg → torch flash → mem-eff
    "sdpa"           → best *torch* kernel (skips the flash_attn package)
    "flash"          → torch SDPA FLASH_ATTENTION (aliases "sdpa_flash")
    "mem_efficient"  → torch SDPA EFFICIENT_ATTENTION (aliases "sdpa_mem_efficient")
    "flash_attn"     → dao-AILab package
    "math"           → exact manual reference
"""

import importlib.metadata
import importlib.util
import math
import os
import warnings
from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Backend constants
# ──────────────────────────────────────────────────────────────────────────────

AUTO = "auto"                        # best available backend
FLASH_ATTN = "flash_attn"            # dao-AILab FlashAttention-2 package
SDPA = "sdpa"                        # best torch SDPA kernel (no package)
SDPA_FLASH = "sdpa_flash"            # torch SDPA FLASH_ATTENTION
SDPA_MEM_EFFICIENT = "sdpa_mem_efficient"  # torch SDPA EFFICIENT_ATTENTION
SDPA_MATH = "sdpa_math"              # torch SDPA MATH (reference fallback)
MATH = "math"                        # exact manual implementation

# User-facing values accepted by config.attn_backend / METIS_ATTN_BACKEND.
USER_FACING_BACKENDS = frozenset(
    {AUTO, FLASH_ATTN, SDPA, "flash", "mem_efficient", MATH}
)
# Aliases that resolve to a concrete SDPA kernel.
_ALIASES = {"flash": SDPA_FLASH, "mem_efficient": SDPA_MEM_EFFICIENT}
# Concrete kernels a call can actually run.
CONCRETE_BACKENDS = frozenset(
    {FLASH_ATTN, SDPA_FLASH, SDPA_MEM_EFFICIENT, SDPA_MATH, MATH}
)

# Fused kernels are fp16/bf16-only and require a multiple-of-8 head dim.
_FUSED_DTYPES = (torch.float16, torch.bfloat16)
_FUSED_MIN_HEAD_DIM, _FUSED_MAX_HEAD_DIM = 8, 256

__all__ = [
    "AUTO", "FLASH_ATTN", "SDPA", "SDPA_FLASH", "SDPA_MEM_EFFICIENT",
    "SDPA_MATH", "MATH", "USER_FACING_BACKENDS",
    "normalize_backend", "detect_attention_backends", "resolve_backend",
    "set_backend_flags", "causal_attention", "math_attention",
    "fused_attention_supported", "_repeat_kv",
]


# ──────────────────────────────────────────────────────────────────────────────
# torch-version-safe SDPA API access
# ──────────────────────────────────────────────────────────────────────────────

_SDP_API = None


def _get_sdpa_api():
    """Return ``(SDPBackend, sdpa_kernel)`` or ``(None, None)``.

    ``sdpa_kernel`` is a module *attribute* (not an importable submodule) on
    torch 2.2+, and ``torch.backends.cuda.sdp_kernel`` on torch 2.0–2.1.
    ``SDPBackend`` is a non-iterable pybind11 enum — members are accessed by
    attribute only, never by iteration.
    """
    global _SDP_API
    if _SDP_API is not None:
        return _SDP_API
    result = (None, None)
    try:
        from torch.nn.attention import SDPBackend

        kernel = getattr(torch.nn.attention, "sdpa_kernel", None)
        if kernel is None:
            kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
        if kernel is not None:
            result = (SDPBackend, kernel)
    except (ImportError, AttributeError):
        try:
            from torch.backends.cuda import SDPBackend, sdp_kernel

            result = (SDPBackend, sdp_kernel)
        except (ImportError, AttributeError):
            result = (None, None)
    _SDP_API = result
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Capability detection (lazy, cached)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BackendCapabilities:
    """One-time snapshot of what attention kernels this machine can run."""

    device: str                      # "cuda" or "cpu"
    torch_version: str
    flash_attn_pkg: str | None    # installed version string, or None
    flash_attn_gqa: bool             # package supports enable_gqa
    torch_flash: bool                # torch SDPA FLASH kernel available (fp16)
    torch_mem_efficient: bool        # torch SDPA EFFICIENT kernel available (fp16)
    torch_math: bool = True
    torch_flash_bf16: bool = False       # FLASH kernel also accepts bf16
    torch_mem_efficient_bf16: bool = False  # EFFICIENT kernel also accepts bf16
    fused_gqa: bool = False          # some fused kernel accepts enable_gqa
    gpu_name: str | None = None      # torch.cuda.get_device_name(0)
    compute_capability: tuple[int, int] | None = None  # e.g. (8, 6) = Ampere
    fused_available: bool = False    # some fused FlashAttention-family kernel works


def _flash_attn_version() -> str | None:
    """Detect the dao-AILab flash-attn package without importing it."""
    if importlib.util.find_spec("flash_attn") is None:
        return None
    for name in ("flash-attn", "flash_attn"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _probe_sdpa_kernel(
    backend_attr: str, dtype: torch.dtype = torch.float16
) -> bool:
    """Run a tiny CUDA call pinned to one SDPA kernel; True if it works.

    ``dtype`` probes per-dtype availability: the fused kernels are
    fp16/bf16-only, and a given build may support only fp16 (e.g. torch's
    memory-efficient kernel on Turing) even when the hardware can compute
    bf16. ``sdpa_kernel`` only allows the listed backend, so an unavailable
    kernel raises instead of silently falling back — exactly what makes this
    a reliable availability probe.
    """
    SDPBackend, sdpa_kernel = _get_sdpa_api()
    if SDPBackend is None or sdpa_kernel is None:
        return False
    kernel = getattr(SDPBackend, backend_attr, None)
    if kernel is None:
        return False
    try:
        q = torch.ones(1, 2, 8, 16, device="cuda", dtype=dtype)
        k = torch.ones(1, 2, 8, 16, device="cuda", dtype=dtype)
        v = torch.ones(1, 2, 8, 16, device="cuda", dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # dispatch emits UserWarnings on purpose
            with sdpa_kernel(kernel):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return True
    except Exception:
        return False


def _probe_fused_gqa() -> bool:
    """True if a fused SDPA kernel natively accepts ``enable_gqa=True``.

    On torch builds where the fused kernels do not broadcast GQA heads
    (e.g. this Windows 2.6.x wheel), the forced call raises and we return
    False, so callers fall back to explicit KV-head expansion.
    """
    doc = getattr(F.scaled_dot_product_attention, "__doc__", "") or ""
    if "enable_gqa" not in doc:
        return False
    SDPBackend, sdpa_kernel = _get_sdpa_api()
    if SDPBackend is None or sdpa_kernel is None:
        return False
    for attr in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION"):
        kernel = getattr(SDPBackend, attr, None)
        if kernel is None:
            continue
        try:
            q = torch.ones(1, 4, 8, 16, device="cuda", dtype=torch.float16)
            k = torch.ones(1, 2, 8, 16, device="cuda", dtype=torch.float16)
            v = torch.ones(1, 2, 8, 16, device="cuda", dtype=torch.float16)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sdpa_kernel(kernel):
                    F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            return True
        except Exception:
            continue
    return False


def _probe_flash_attn_gqa() -> bool:
    """True if the installed flash-attn package accepts ``enable_gqa``."""
    if _flash_attn_version() is None:
        return False
    try:
        import flash_attn

        fn = flash_attn.flash_attn_func
        params = set()
        try:
            import inspect

            params = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = set()
        if "enable_gqa" not in params:
            return False
        q = torch.ones(1, 8, 4, 16, device="cuda", dtype=torch.float16)
        k = torch.ones(1, 8, 2, 16, device="cuda", dtype=torch.float16)
        v = torch.ones(1, 8, 2, 16, device="cuda", dtype=torch.float16)
        fn(q, k, v, causal=True, enable_gqa=True)
        return True
    except Exception:
        return False


@cache
def _probe() -> BackendCapabilities:
    """Build the (cached) capability snapshot for this process / GPU.

    Kernel availability is probed empirically (a real tiny CUDA call pinned to
    each kernel), which is more reliable than inferring from the compute
    capability. The compute capability / device name are reported alongside as
    informational GPU capability fields.
    """
    cuda = torch.cuda.is_available()
    torch_flash = _probe_sdpa_kernel("FLASH_ATTENTION") if cuda else False
    torch_mem_efficient = _probe_sdpa_kernel("EFFICIENT_ATTENTION") if cuda else False
    flash_attn_pkg = _flash_attn_version() if cuda else None
    return BackendCapabilities(
        device="cuda" if cuda else "cpu",
        torch_version=torch.__version__,
        flash_attn_pkg=flash_attn_pkg,
        flash_attn_gqa=_probe_flash_attn_gqa() if cuda else False,
        torch_flash=torch_flash,
        torch_mem_efficient=torch_mem_efficient,
        torch_math=True,
        torch_flash_bf16=(
            _probe_sdpa_kernel("FLASH_ATTENTION", dtype=torch.bfloat16) if cuda else False
        ),
        torch_mem_efficient_bf16=(
            _probe_sdpa_kernel("EFFICIENT_ATTENTION", dtype=torch.bfloat16) if cuda else False
        ),
        fused_gqa=_probe_fused_gqa() if cuda else False,
        gpu_name=torch.cuda.get_device_name(0) if cuda else None,
        compute_capability=tuple(torch.cuda.get_device_capability(0)) if cuda else None,
        fused_available=(flash_attn_pkg is not None or torch_flash or torch_mem_efficient),
    )


def _fused_dtype_supported(
    concrete: str, dtype: torch.dtype, cap: BackendCapabilities
) -> bool:
    """True if the fused SDPA kernel ``concrete`` accepts ``dtype``.

    fp16 is supported by every fused build; bf16 support varies by build and
    GPU (torch's memory-efficient kernel historically lacks it on Turing,
    where ``torch.cuda.is_bf16_supported()`` still reports True). The
    dao-AILab flash-attn package (``FLASH_ATTN``) accepts fp16 and bf16.
    """
    if dtype != torch.bfloat16:
        return True
    if concrete == SDPA_FLASH:
        return cap.torch_flash_bf16
    if concrete == SDPA_MEM_EFFICIENT:
        return cap.torch_mem_efficient_bf16
    return True  # FLASH_ATTN (package) supports bf16


def _auto_concrete(cap: BackendCapabilities, dtype: torch.dtype | None = None) -> str:
    """The concrete kernel ``auto`` resolves to on this machine.

    ``dtype`` (default: fp16 — the probe's dtype) narrows the choice to
    kernels that actually accept the call's dtype, so bf16 on a Turing GPU
    (fused kernels are fp16-only there) degrades to torch's SDPA math kernel
    instead of crashing with "No available kernel".
    """
    dtype = dtype if dtype is not None else torch.float16
    if cap.device == "cuda":
        if cap.flash_attn_pkg is not None:
            return FLASH_ATTN
        if cap.torch_flash and _fused_dtype_supported(SDPA_FLASH, dtype, cap):
            return SDPA_FLASH
        if cap.torch_mem_efficient and _fused_dtype_supported(SDPA_MEM_EFFICIENT, dtype, cap):
            return SDPA_MEM_EFFICIENT
        return SDPA_MATH  # torch's SDPA math kernel accepts any dtype on CUDA
    return MATH


def detect_attention_backends() -> dict:
    """Return a human/benchmark-friendly report of this machine's kernels.

    Includes the detected GPU capability (device name + compute capability, e.g.
    ``(8, 6)`` for an Ampere sm_86 part) alongside per-kernel availability.
    ``recommended`` is what ``backend="auto"`` resolves to for a typical
    fp16/bf16 CUDA call. Cheap and safe on CPU (cached probe).
    """
    cap = _probe()
    return {
        "device": cap.device,
        "gpu_name": cap.gpu_name,
        "compute_capability": cap.compute_capability,
        "torch": cap.torch_version,
        "flash_attn": cap.flash_attn_pkg,
        "flash_attn_gqa": cap.flash_attn_gqa,
        "torch_flash": cap.torch_flash,
        "torch_mem_efficient": cap.torch_mem_efficient,
        "torch_flash_bf16": cap.torch_flash_bf16,
        "torch_mem_efficient_bf16": cap.torch_mem_efficient_bf16,
        "torch_math": cap.torch_math,
        "fused_gqa": cap.fused_gqa,
        "fused_available": cap.fused_available,
        "recommended": _auto_concrete(cap),
    }


def fused_attention_supported() -> bool:
    """True when this machine can run a fused FlashAttention-family kernel.

    Covers, in priority order: the dao-AILab ``flash_attn`` package, torch SDPA
    FLASH_ATTENTION, and torch SDPA EFFICIENT_ATTENTION (the memory-efficient
    online-softmax kernel). On CPU (or pre-Ampere GPUs) this is ``False`` and
    the dispatcher automatically degrades to the exact manual reference, so
    callers never need to branch on this.
    """
    return _probe().fused_available


# ──────────────────────────────────────────────────────────────────────────────
# Backend resolution
# ──────────────────────────────────────────────────────────────────────────────

_warned = set()


def _warn_unavailable(requested: str, reason: str = "") -> None:
    """Warn once per (backend, reason) that a forced kernel fell back."""
    key = (requested, reason)
    if key in _warned:
        return
    _warned.add(key)
    detail = f" ({reason})" if reason else ""
    warnings.warn(
        f"Attention backend {requested!r} is not available{detail}; "
        f"falling back to the best available backend.",
        stacklevel=3,
    )


def _warn_sdpa_fallback(backend: str, exc: Exception) -> None:
    """Warn once that a selected kernel failed at call time."""
    key = ("sdpa_runtime", backend)
    if key in _warned:
        return
    _warned.add(key)
    warnings.warn(
        f"Attention backend {backend!r} failed at runtime ({exc}); "
        f"falling back to the exact manual math reference.",
        stacklevel=3,
    )


def normalize_backend(requested: str | None) -> str:
    """Validate a user-facing backend name and resolve aliases.

    Returns one of ``AUTO``, ``FLASH_ATTN``, ``SDPA``, ``SDPA_FLASH``,
    ``SDPA_MEM_EFFICIENT``, ``MATH``. Raises ``ValueError`` on unknown names.
    """
    if requested is None:
        return AUTO
    if requested in _ALIASES:
        return _ALIASES[requested]
    if requested in USER_FACING_BACKENDS:
        return requested
    raise ValueError(
        f"Unknown attention backend {requested!r}. "
        f"Allowed: {sorted(USER_FACING_BACKENDS)}"
    )


def resolve_backend(
    requested: str,
    q: torch.Tensor,
    *,
    cap: BackendCapabilities | None = None,
    use_flash_attn: bool = True,
) -> str:
    """Resolve the concrete kernel for one call. Pure and data-independent.

    Precedence: ``METIS_ATTN_BACKEND`` env var > explicit ``requested`` (when
    not ``auto``) > ``use_flash_attn`` (``False`` → ``math``).

    The result depends only on the env var, the request, the capabilities
    snapshot, and the tensors' dtype / device / head dim — never on tensor
    values — so ``torch.compile`` can specialize the branch.
    """
    cap = cap if cap is not None else _probe()

    env = os.environ.get("METIS_ATTN_BACKEND")
    if env:
        requested = normalize_backend(env)
    else:
        requested = normalize_backend(requested)
        if requested == AUTO and not use_flash_attn:
            requested = MATH

    if requested == MATH:
        return MATH

    if q.device.type != "cuda":
        if requested not in (AUTO, SDPA):
            _warn_unavailable(requested, "CPU")
        return MATH

    dtype = q.dtype
    head_dim = q.size(-1)
    fused_eligible = (
        dtype in _FUSED_DTYPES
        and _FUSED_MIN_HEAD_DIM <= head_dim <= _FUSED_MAX_HEAD_DIM
        and head_dim % 8 == 0
    )
    if not fused_eligible:
        if requested not in (AUTO, SDPA):
            _warn_unavailable(requested, f"dtype {dtype}")
        return MATH

    # Fused-kernel selection is dtype-aware: a kernel that is compiled but
    # rejects this dtype (bf16 on builds whose fused kernels are fp16-only)
    # is skipped rather than selected and left to crash at call time.
    if requested == FLASH_ATTN:
        if cap.flash_attn_pkg is not None:
            return FLASH_ATTN
        _warn_unavailable(requested)
        return _auto_concrete(cap, dtype)
    if requested == SDPA_FLASH:
        if cap.torch_flash and _fused_dtype_supported(SDPA_FLASH, dtype, cap):
            return SDPA_FLASH
        _warn_unavailable(requested, f"dtype {dtype}")
        return _auto_concrete(cap, dtype)
    if requested == SDPA_MEM_EFFICIENT:
        if cap.torch_mem_efficient and _fused_dtype_supported(SDPA_MEM_EFFICIENT, dtype, cap):
            return SDPA_MEM_EFFICIENT
        _warn_unavailable(requested, f"dtype {dtype}")
        return _auto_concrete(cap, dtype)
    if requested == SDPA:
        if cap.torch_flash and _fused_dtype_supported(SDPA_FLASH, dtype, cap):
            return SDPA_FLASH
        if cap.torch_mem_efficient and _fused_dtype_supported(SDPA_MEM_EFFICIENT, dtype, cap):
            return SDPA_MEM_EFFICIENT
        return SDPA_MATH if cap.device == "cuda" else MATH
    return _auto_concrete(cap, dtype)  # AUTO


def set_backend_flags(backend: str = AUTO) -> None:
    """Configure the process-global torch SDPA kernel flags.

    Called once at model construction (not in the hot path). For ``auto`` /
    ``sdpa`` / ``math`` all SDPA kernels stay enabled (auto-dispatch). For a
    concrete fused backend only that kernel is enabled. The math kernel is
    always left enabled as a crash safety net. No-op on non-CUDA builds.

    The flags are process-global: constructing a second model with a different
    backend overrides the flags for the first. The per-call dispatcher uses an
    explicit ``sdpa_kernel`` scope (authoritative) outside of ``torch.compile``,
    so this only matters under compilation or for direct SDPA users.
    """
    if not torch.cuda.is_available():
        return
    bc = torch.backends.cuda
    set_flash = getattr(bc, "enable_flash_sdp", None)
    set_mem = getattr(bc, "enable_mem_efficient_sdp", None)
    set_math = getattr(bc, "enable_math_sdp", None)
    req = normalize_backend(backend)
    if req == SDPA_FLASH and set_flash and set_mem:
        set_flash(True)
        set_mem(False)
    elif req == SDPA_MEM_EFFICIENT and set_flash and set_mem:
        set_flash(False)
        set_mem(True)
    else:  # auto / sdpa / flash_attn / math — keep auto-dispatch intact
        if set_flash:
            set_flash(True)
        if set_mem:
            set_mem(True)
    if set_math:
        set_math(True)


# ──────────────────────────────────────────────────────────────────────────────
# Core kernels
# ──────────────────────────────────────────────────────────────────────────────

def _repeat_kv(x: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Repeat KV heads to match query heads (GQA → MHA)."""
    if n_groups == 1:
        return x
    B, n_kv, T, head_dim = x.size()
    x = x[:, :, None, :, :].expand(B, n_kv, n_groups, T, head_dim)
    return x.reshape(B, n_kv * n_groups, T, head_dim)


def _normalize_attention_mask(
    mask: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Coerce an attention mask to ``(B, 1, T_q, T_k)`` bool.

    Accepts ``(B, T_q, T_k)`` or ``(B, 1, T_q, T_k)`` in any numeric or bool
    dtype; ``True`` marks positions that *may* attend (causal + same-segment),
    matching SDPA's ``attn_mask`` convention.
    """
    B, T_q, _ = q.size(0), q.size(2), q.size(3)
    T_k = mask.size(-1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() == 4:
        if mask.size(1) != 1 and mask.size(1) != q.size(1):
            raise ValueError(
                f"attention_mask head dim must be 1 or {q.size(1)}, "
                f"got {mask.size(1)}"
            )
    else:
        raise ValueError(
            f"attention_mask must be (B, T_q, T_k) or (B, 1, T_q, T_k), "
            f"got shape {tuple(mask.shape)}"
        )
    if mask.size(0) != B or mask.size(2) != T_q or mask.size(3) != T_k:
        raise ValueError(
            f"attention_mask shape {tuple(mask.shape)} incompatible with "
            f"Q (B={B}, T_q={T_q}, T_k={T_k})"
        )
    return mask.bool()


def math_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: float | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact manual causal attention — the legacy reference implementation.

    Byte-identical to the old ``CausalSelfAttention`` manual path (``q @ kᵀ``,
    ``masked_fill(-inf)`` over the lower triangle, softmax, dropout, ``@ v``).
    ``k`` / ``v`` must already be expanded to ``n_heads`` (callers use
    :func:`_repeat_kv` for GQA). ``T_q == 1`` decode gets no masking, matching
    the legacy behavior.

    Args:
        q: (B, n_heads, T_q, head_dim)
        k: (B, n_heads, T_k, head_dim)
        v: (B, n_heads, T_k, head_dim)
        dropout_p: dropout applied to attention weights (0 disables it).
        is_causal: mask the upper triangle with ``-inf``. When ``T_q < T_k``
            (decoder decode against a prefix cache) the mask is the extended
            lower triangle: every query attends to every cached key — this
            exactly reproduces the legacy ``CausalSelfAttention`` buffer slice.
        scale: softmax temperature (``1/sqrt(head_dim)`` if None).
        attention_mask: optional ``(B, 1, T_q, T_k)`` bool mask (block-diagonal
            causal for packed training). When given it is authoritative and
            ``is_causal`` is ignored.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    att = (q @ k.transpose(-2, -1)) * scale
    T_q, T_k = q.size(2), k.size(2)
    if attention_mask is not None:
        mask = _normalize_attention_mask(attention_mask, q)
        if mask.size(-2) != T_q or mask.size(-1) != T_k:
            # Allow a mask built for T_q == T_k (prefill) to be reused when a
            # decode prefix cache extends T_k; treat missing columns as blocked.
            full = torch.zeros(1, 1, T_q, T_k, dtype=torch.bool, device=q.device)
            full[..., : mask.size(-2), : mask.size(-1)] = mask
            mask = full
        att = att.masked_fill(~mask, float("-inf"))
    elif is_causal:
        # diagonal = T_k - T_q reproduces the legacy sliced tril(max_seq_len)
        # buffer: prefill (T_q == T_k) → plain tril; decode (T_q < T_k) →
        # row i attends to columns 0..(T_k - T_q + i), i.e. all cached keys.
        mask = torch.tril(
            torch.ones(T_q, T_k, device=q.device), diagonal=T_k - T_q
        ).view(1, 1, T_q, T_k)
        att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att = F.dropout(att, p=dropout_p, training=dropout_p > 0)
    return att @ v


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def _maybe_scale_q(q: torch.Tensor, scale: float) -> torch.Tensor:
    """SDPA hard-codes ``1/sqrt(head_dim)``; fold any other scale into q."""
    default = 1.0 / math.sqrt(q.size(-1))
    if abs(scale - default) < 1e-12:
        return q
    return q * (scale * math.sqrt(q.size(-1)))


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    is_causal: bool,
    backend: str,
    attention_mask: torch.Tensor | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Run ``F.scaled_dot_product_attention`` pinned to ``backend``.

    Outside ``torch.compile`` an explicit ``sdpa_kernel`` scope makes the
    selection authoritative regardless of the global flags. Under compilation
    the scope cannot be traced, so we rely on the global flags (set by
    :func:`set_backend_flags` to match).

    ``attention_mask`` (bool, ``(B, 1, T_q, T_k)``) is passed through as
    ``attn_mask``; when present ``is_causal`` is dropped because the mask is
    already block-diagonal causal.

    ``enable_gqa=True`` hands ``F.scaled_dot_product_attention`` unexpanded KV
    heads (``n_kv_heads < n_heads``) and lets the fused kernels broadcast them
    natively. Only legal when the resolved kernel actually supports GQA — the
    caller gates this on ``cap.fused_gqa``.
    """
    SDPBackend, sdpa_kernel = _get_sdpa_api()
    kwargs = dict(
        query=q, key=k, value=v, dropout_p=dropout_p, is_causal=is_causal
    )
    if enable_gqa:
        kwargs["enable_gqa"] = True
    if attention_mask is not None:
        kwargs["attn_mask"] = attention_mask
        kwargs["is_causal"] = False
    kernel = None
    if SDPBackend is not None:
        if backend == SDPA_FLASH:
            kernel = getattr(SDPBackend, "FLASH_ATTENTION", None)
        elif backend == SDPA_MEM_EFFICIENT:
            kernel = getattr(SDPBackend, "EFFICIENT_ATTENTION", None)
        elif backend == SDPA_MATH:
            # Pinning MATH suppresses the per-call "fused kernel not used"
            # warnings that auto-dispatch emits for a dtype the fused kernels
            # reject (e.g. bf16 on Turing) and guarantees the call succeeds.
            kernel = getattr(SDPBackend, "MATH", None)
    compiling = getattr(
        getattr(torch, "compiler", None), "is_compiling", lambda: False
    )()
    if kernel is not None and sdpa_kernel is not None and not compiling:
        with sdpa_kernel(kernel):
            return F.scaled_dot_product_attention(**kwargs)
    return F.scaled_dot_product_attention(**kwargs)


def _flash_attn_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
    scale: float | None,
    groups: int,
    cap: BackendCapabilities,
) -> torch.Tensor | None:
    """Run the dao-AILab FlashAttention-2 kernel. ``None`` on any failure."""
    try:
        import flash_attn
    except ImportError:
        return None
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    qq = q.transpose(1, 2).contiguous()  # (B, T, H, D) — the package's layout
    if groups > 1 and cap.flash_attn_gqa:
        kk = k.transpose(1, 2).contiguous()
        vv = v.transpose(1, 2).contiguous()
        out = flash_attn.flash_attn_func(
            qq, kk, vv, dropout_p=dropout_p,
            softmax_scale=scale, causal=causal, enable_gqa=True,
        )
    else:
        kk = _repeat_kv(k, groups).transpose(1, 2).contiguous()
        vv = _repeat_kv(v, groups).transpose(1, 2).contiguous()
        out = flash_attn.flash_attn_func(
            qq, kk, vv, dropout_p=dropout_p,
            softmax_scale=scale, causal=causal,
        )
    return out.transpose(1, 2)


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = True,
    scale: float | None = None,
    n_heads: int | None = None,
    n_kv_heads: int | None = None,
    backend: str = AUTO,
    use_flash_attn: bool = True,
    training: bool = False,
    out_backend: list | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dispatch causal attention to the best available backend.

    Args:
        q: (B, n_heads, T_q, head_dim) query heads.
        k: (B, n_kv_heads, T_k, head_dim) — *unexpanded* KV heads.
        v: (B, n_kv_heads, T_k, head_dim).
        dropout_p: attention dropout (applied by the kernel / math path).
        is_causal: causal masking. ``T_q == 1`` decode needs no explicit mask.
        scale: softmax temperature (``1/sqrt(head_dim)`` if None).
        n_heads: query head count (from ``q.size(1)`` if omitted).
        n_kv_heads: KV head count; GQA groups = ``n_heads // n_kv_heads``.
        backend: user-facing selection (see module docstring).
        use_flash_attn: ``False`` pins the exact manual math reference.
        training: propagate to kernels that need to know (dropout semantics).
        out_backend: if a list, appends the concrete backend actually used.
        attention_mask: optional ``(B, 1, T_q, T_k)`` bool mask for packed
            training (block-diagonal causal). When provided it is authoritative
            (``is_causal`` is ignored) and the dao-AILab flash-attn package is
            skipped — FA2 has no mask support — degrading to torch SDPA or the
            math reference.

    Returns:
        y: (B, n_heads, T_q, head_dim). Identical shape in every backend;
        numerical differences between backends are within fused-kernel
        rounding tolerance (fp32 accumulation inside flash/mem-eff).
    """
    head_dim = q.size(-1)
    if n_heads is None:
        n_heads = q.size(1)
    groups = max(1, n_heads // n_kv_heads) if n_kv_heads else 1

    mask = None
    if attention_mask is not None:
        mask = _normalize_attention_mask(attention_mask, q)

    # Causal masking applies to prefill (T_q == T_k). During decode the cache
    # is the strict prefix of the sequence, so every new query legitimately
    # attends to every cached key — masking (and torch's is_causal with
    # T_q < T_k, which mis-masks on this torch build) must be avoided.
    T_q, T_k = q.size(2), k.size(2)
    effective_causal = bool(is_causal and T_q >= T_k and mask is None)

    cap = _probe()
    concrete = resolve_backend(
        backend, q, cap=cap, use_flash_attn=use_flash_attn
    )

    if mask is not None and concrete == FLASH_ATTN:
        # dao-AILab FA2 accepts only causal (no arbitrary mask) — fall back to a
        # torch SDPA kernel that does (and accepts this dtype), or to math.
        if cap.torch_mem_efficient and _fused_dtype_supported(
            SDPA_MEM_EFFICIENT, q.dtype, cap
        ):
            concrete = SDPA_MEM_EFFICIENT
        elif cap.device == "cuda":
            concrete = SDPA_MATH
        else:
            concrete = MATH

    if concrete == FLASH_ATTN:
        y = _flash_attn_2(
            q, k, v,
            dropout_p=dropout_p if training else 0.0,
            causal=effective_causal, scale=scale, groups=groups, cap=cap,
        )
        if y is not None:
            if out_backend is not None:
                out_backend.append(FLASH_ATTN)
            return y
        concrete = _auto_concrete(cap, q.dtype)
        if concrete == FLASH_ATTN:  # package present but call failed — guard
            concrete = MATH

    if concrete in (SDPA_FLASH, SDPA_MEM_EFFICIENT, SDPA_MATH):
        # GQA is only passed through to the fused kernels natively; the math
        # kernel always receives explicitly expanded KV heads.
        use_gqa = (
            groups > 1 and cap.fused_gqa
            and concrete in (SDPA_FLASH, SDPA_MEM_EFFICIENT)
        )
        try:
            if use_gqa:
                # Unexpanded KV heads + enable_gqa: the fused kernel broadcasts
                # n_kv_heads → n_heads internally (no _repeat_kv allocation).
                # Without the flag the kernel would see mismatched head counts
                # and throw, silently degrading every GQA call to the math path.
                y = _sdpa(
                    _maybe_scale_q(q, scale or 1.0 / math.sqrt(head_dim)),
                    k, v,
                    dropout_p=dropout_p, is_causal=effective_causal,
                    backend=concrete, attention_mask=mask,
                    enable_gqa=True,
                )
            else:
                k2 = _repeat_kv(k, groups)
                v2 = _repeat_kv(v, groups)
                y = _sdpa(
                    _maybe_scale_q(q, scale or 1.0 / math.sqrt(head_dim)),
                    k2, v2,
                    dropout_p=dropout_p, is_causal=effective_causal,
                    backend=concrete, attention_mask=mask,
                )
        except RuntimeError as exc:
            # Last-resort safety net: a kernel that slipped past resolution
            # rejected the call (e.g. "No available kernel" on a dtype/build
            # mismatch). Degrade to the exact math reference — never crash.
            _warn_sdpa_fallback(concrete, exc)
            concrete = MATH
        else:
            if out_backend is not None:
                out_backend.append(concrete)
            return y

    # Math (reference) path.
    k2 = _repeat_kv(k, groups)
    v2 = _repeat_kv(v, groups)
    y = math_attention(
        q, k2, v2, dropout_p=dropout_p, is_causal=effective_causal,
        scale=scale, attention_mask=mask,
    )
    if out_backend is not None:
        out_backend.append(MATH)
    return y
