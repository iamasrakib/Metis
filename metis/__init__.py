"""
Μῆτις (Metis) — A modern tiny language model, built from scratch in PyTorch.

Public API:

    from metis import MetisLM, ModelConfig, CharTokenizer, generate_text

    config = ModelConfig.from_preset("medium", max_iters=10000)
    model  = MetisLM(config)
    out    = generate_text(model, tokenizer, "Once upon a time")
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .attn import (
    AUTO,
    FLASH_ATTN,
    MATH,
    SDPA,
    SDPA_FLASH,
    SDPA_MATH,
    SDPA_MEM_EFFICIENT,
    causal_attention,
    detect_attention_backends,
    fused_attention_supported,
    math_attention,
    normalize_backend,
    resolve_backend,
    set_backend_flags,
)

# Core public objects ─────────────────────────────────────────────────────────
from .config import (
    PRESETS,
    ModelConfig,
    setup_logging,
)
from .data import (
    BPETokenizer,
    CharTokenizer,
    MMapDataset,
    StreamingTextDataset,
    TextDataset,
    create_dataloader,
    get_dataloader,  # backward-compatible alias
    load_text,
    train_val_split,
)
from .expert_cache import (
    expert_cache_bandwidth_reduction,
    expert_cache_hit_rate,
)
from .generate import (
    chat,
    generate_text,
    load_model_and_tokenizer,
)
from .kv import (  # KV cache subsystem (Phase 7) — public re-export
    KVBackendInfo,
    KVCache,
    LayerKV,
    cache_memory_bytes,
    cached_bytes,
    cached_len_of,
    dequantize_per_token,
    kv_cache_ratio,
    quantize_per_token,
)
from .mla import (  # public re-export
    MLAAttention,
    MLALayerCache,
)
from .model import (
    MLP,
    CausalSelfAttention,
    MetisLM,
    RMSNorm,
    SwiGLU,
    TinyLLM,  # backward-compatible alias
    TransformerBlock,
)
from .moe import (
    GROUPED,
    PER_EXPERT,
    ExpertCache,
    MoE,
    detect_moe_engines,
    forward_grouped,
    forward_per_expert,
    resolve_engine,
)
from .pipeline import (
    AsyncCheckpointer,
    GpuBatchStager,
    GpuIdleTracker,
    ThreadPrefetcher,
)
from .scheduler import (
    INFER,
    TRAIN,
    ExecutionPlan,
    ExecutionScheduler,
    build_scheduler,
    plan_execution,
)
from .training import (
    estimate_loss,
    get_lr,
    load_checkpoint,
    save_checkpoint,
    train,
)

try:
    __version__ = _version("metis-llm")
except PackageNotFoundError:  # not installed (e.g. running from source)
    __version__ = "3.0.0"

__all__ = [
    # version
    "__version__",
    # config
    "ModelConfig",
    "PRESETS",
    "setup_logging",
    # data
    "BPETokenizer",
    "CharTokenizer",
    "TextDataset",
    "MMapDataset",
    "StreamingTextDataset",
    "load_text",
    "train_val_split",
    "create_dataloader",
    "get_dataloader",
    # model
    "MetisLM",
    "TinyLLM",
    "RMSNorm",
    "SwiGLU",
    "MLP",
    "CausalSelfAttention",
    "TransformerBlock",
    # MoE
    "MoE",
    "ExpertCache",
    "GROUPED",
    "PER_EXPERT",
    "forward_grouped",
    "forward_per_expert",
    "resolve_engine",
    "detect_moe_engines",
    "expert_cache_hit_rate",
    "expert_cache_bandwidth_reduction",
    # attention
    "AUTO",
    "FLASH_ATTN",
    "MATH",
    "SDPA",
    "SDPA_FLASH",
    "SDPA_MATH",
    "SDPA_MEM_EFFICIENT",
    "causal_attention",
    "detect_attention_backends",
    "fused_attention_supported",
    "math_attention",
    "normalize_backend",
    "resolve_backend",
    "set_backend_flags",
    # generate
    "generate_text",
    "chat",
    "load_model_and_tokenizer",
    # train
    "train",
    "get_lr",
    "estimate_loss",
    "save_checkpoint",
    "load_checkpoint",
    # overlapped pipeline
    "ThreadPrefetcher",
    "GpuBatchStager",
    "AsyncCheckpointer",
    "GpuIdleTracker",
    # exec scheduler
    "build_scheduler",
    "ExecutionScheduler",
    "ExecutionPlan",
    "plan_execution",
    "INFER",
    "TRAIN",
    # KV cache subsystem (Phase 7)
    "KVBackendInfo",
    "KVCache",
    "LayerKV",
    "cache_memory_bytes",
    "cached_bytes",
    "cached_len_of",
    "kv_cache_ratio",
    "quantize_per_token",
    "dequantize_per_token",
    "MLAAttention",
    "MLALayerCache",
]
