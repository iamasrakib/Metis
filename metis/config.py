"""
Μῆτις (Metis) — Configuration System
======================================
Centralized configuration with validation, CLI overrides, preset profiles,
and structured logging. All hyperparameters are documented and type-checked.
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_dir: str | None = None) -> logging.Logger:
    """Configure project-wide logging with console + optional file output."""
    logger = logging.getLogger("metis")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    # Console handler with color
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(console)

    # File handler (if log_dir provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "metis.log"), encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Model Presets
# ──────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "tiny": {
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "max_seq_len": 256,
        "description": "~1M params — fastest training, proof-of-concept",
    },
    "small": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "max_seq_len": 256,
        "description": "~4M params — good balance for ≤6 GB VRAM",
    },
    "medium": {
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 6,
        "max_seq_len": 512,
        "description": "~15M params — richer capacity, needs ≥6 GB VRAM",
    },
    "large": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 8,
        "max_seq_len": 512,
        "description": "~35M params — full capacity, needs ≥8 GB VRAM",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Central configuration for the Metis language model.

    Covers model architecture, training hyperparameters, data paths, and
    runtime settings. Use ``ModelConfig.from_preset()`` for quick setup
    or ``ModelConfig.from_json()`` to load a saved configuration.
    """

    # ── Architecture ──────────────────────────────────────────────────────
    vocab_size: int = 256               # Updated at runtime from tokenizer
    d_model: int = 256                  # Embedding / hidden dimension
    n_heads: int = 4                    # Number of attention heads
    n_kv_heads: int = 0                 # KV heads for GQA (0 = use n_heads = MHA)
    n_layers: int = 4                   # Number of transformer blocks
    max_seq_len: int = 256              # Maximum context length
    dropout: float = 0.1               # Dropout rate (0 = disabled)
    use_rmsnorm: bool = True            # Use RMSNorm instead of LayerNorm
    use_swiglu: bool = True             # Use SwiGLU instead of GELU MLP
    use_rope: bool = True               # Use Rotary Position Embeddings
    tie_weights: bool = True            # Tie embedding & output weights
    # ── Architecture: MoE (Phase 3) ──────────────────────────────────────
    use_moe: bool = False               # Enable Mixture of Experts
    moe_num_experts: int = 8            # Number of experts in MoE layer
    moe_top_k: int = 2                  # Top-k experts per token
    moe_engine: str = "auto"            # "auto"|"grouped"|"per_expert" (see metis/moe.py)
    moe_group_ratio: float = 2.0        # Max/min token ratio inside one expert
                                        # group; lower = tighter, more bmm pairs
                                        # (see metis/moe.py)
    moe_cache_size: int = 64            # Persistent expert cache: max group entries
                                        # (0 = disabled; see metis/expert_cache.py)
    moe_cache_bytes: int = 0            # Optional byte budget for the expert cache
                                        # (0 = unbounded by bytes)
    # ── Architecture: QK-Normalization (Phase 3) ─────────────────────────
    use_qk_norm: bool = False           # Enable query/key normalization
    # ── Architecture: Attention Sink (Phase 3) ───────────────────────────
    use_attention_sink: bool = False    # Enable attention sink for long context

    # ── Training ──────────────────────────────────────────────────────────
    micro_batch_size: int = 8           # Samples per forward pass
    gradient_accumulation_steps: int = 8  # Effective batch = micro × accum
    max_iters: int = 5000               # Total training iterations
    learning_rate: float = 3e-4         # Peak learning rate
    min_lr: float = 1e-5                # Minimum learning rate (cosine floor)
    weight_decay: float = 0.1           # AdamW weight decay
    warmup_steps: int = 500             # Linear warmup steps
    max_grad_norm: float = 1.0          # Gradient clipping norm
    val_interval: int = 100             # Validate every N iterations
    val_steps: int = 20                 # Batches per validation
    sample_interval: int = 200          # Generate sample every N iterations
    save_interval: int = 500            # Checkpoint every N iterations
    log_interval: int = 10              # Log metrics every N iterations
    train_split: float = 0.9            # Train/validation split ratio
    # ── Training: EMA (Phase 2) ──────────────────────────────────────────
    use_ema: bool = False               # Exponential Moving Average of weights
    ema_decay: float = 0.999            # EMA decay rate
    # ── Training: DDP (Phase 2) ──────────────────────────────────────────
    use_ddp: bool = False               # Distributed Data Parallel
    ddp_world_size: int = 1             # Number of DDP processes

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "data"
    dataset_path: str = "data/input.txt"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    # ── Data: Tokenizer (Phase 1) ────────────────────────────────────────
    tokenizer: str = "char"             # Tokenizer: "char" | "cl100k_base" | "p50k_base" | ...
    use_mmap: bool = True               # Use memory-mapped dataset (Phase 1)
    num_workers: int = 0                # DataLoader worker processes
    force_recache: bool = False         # Force re-tokenization cache rebuild
    # ── Data: Dynamic Sequence Packing (Phase 5) ─────────────────────────
    use_packing: bool = False           # Pack short sequences into dense fixed-length batches
    packing_strategy: str = "stream"    # "stream" (contiguous, zero-pad) | "bin" (whole-doc FFD)

    # ── Runtime ───────────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    compile_model: bool = False         # torch.compile (PyTorch 2.0+)
    use_cuda_graphs: bool = True        # CUDA Graphs for the training step
    # ── Training: Overlapped pipeline (Phase 6) ────────────────────────────
    use_pipeline: bool = True           # Prefetch + async H2D + async checkpoints
    prefetch_depth: int = 2             # Steps read ahead by the prefetch thread
    async_checkpoint: bool = True       # Write checkpoints on a background thread
    pipeline_buffer_depth: int = 3      # H2D staging ring depth (copy-stream slack)
    use_layer_prefetch: bool = True     # Warm the next layer's expert cache during compute
    seed: int = 42                      # Reproducibility seed
    log_level: str = "INFO"             # Logging verbosity

    # ── Experiment Tracking (Phase 2) ─────────────────────────────────────
    use_wandb: bool = False             # Log metrics to Weights & Biases
    wandb_project: str = "metis-llm"    # W&B project name
    wandb_run_name: str = ""            # W&B run name (auto-generated if empty)

    # ── Inference (Phase 4) ───────────────────────────────────────────────
    quantize: str = "none"              # Quantization: "none" | "int8" | "fp8"
    use_flash_attn: bool = True         # Use Flash Attention v2 when available
    attn_backend: str = "auto"          # "auto"|"flash_attn"|"sdpa"|"flash"|"mem_efficient"|"math"
    use_exec_scheduler: bool = False    # Graph-based execution scheduler (inference)

    # ── Inference: KV cache subsystem (Phase 7) ───────────────────────────
    kv_backend: str = "default"         # KV cache engine (see metis/kv.py):
                                        #   "default"   — legacy growable list of (K, V) tuples
                                        #                 appended via torch.cat (reference)
                                        #   "static"    — preallocated contiguous buffers,
                                        #                 in-place writes, flat memory
                                        #   "quantized" — static layout + int8 compressed
                                        #                 cache (per-token scales)
                                        #   "mla"       — Multi-head Latent Attention
                                        #                 (architecture change — train from
                                        #                 scratch; see metis/mla.py)
    kv_cache_dtype: str = "auto"        # Cache element dtype for the "static" backend:
                                        #   "auto" = compute dtype | "fp32" | "fp16" | "bf16"
                                        # (fp16/bf16 halve cache memory with negligible error)
    kv_quant_scheme: str = "int8"       # Quantized-cache dtype for kv_backend="quantized"
                                        #   "int8" — per-token symmetric, ~4x memory cut
    # ── Inference: Multi-head Latent Attention (Phase 7) ──────────────────
    mla_kv_latent_dim: int = 0          # MLA latent KV dim c_d (0 = d_model // n_heads).
                                        # Shared across heads; larger = more expressive.
    mla_rope_head_dim: int = 0          # MLA RoPE part of the key head dim (0 = head_dim // 2).
                                        # Must be even (RoPE works on pairs).
    mla_scale_head_dim: bool = False    # Scale attention logits by sqrt(content + rope dims)
                                        # (False = sqrt(content dim only), the DeepSeek choice)

    # ── Derived (computed in __post_init__) ────────────────────────────────
    head_dim: int = field(init=False)
    effective_batch_size: int = field(init=False)
    n_params: str = field(init=False, default="unknown")

    @property
    def n_groups(self) -> int:
        """Query groups per KV head (n_heads // n_kv_heads)."""
        return self.n_heads // self.n_kv_heads

    def __post_init__(self):
        """Validate configuration and create directories."""
        # Validation
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_kv_heads == 0:
            self.n_kv_heads = self.n_heads  # default = MHA
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0, got {self.max_grad_norm}")
        _ATTN_BACKENDS = ("auto", "flash_attn", "sdpa", "flash", "mem_efficient", "math")
        if self.attn_backend not in _ATTN_BACKENDS:
            raise ValueError(
                f"attn_backend must be one of {_ATTN_BACKENDS}, "
                f"got {self.attn_backend!r}"
            )
        _KV_BACKENDS = ("default", "static", "quantized", "mla")
        if self.kv_backend not in _KV_BACKENDS:
            raise ValueError(
                f"kv_backend must be one of {_KV_BACKENDS}, got {self.kv_backend!r}"
            )
        _KV_CACHE_DTYPES = ("auto", "fp32", "fp16", "bf16")
        if self.kv_cache_dtype not in _KV_CACHE_DTYPES:
            raise ValueError(
                f"kv_cache_dtype must be one of {_KV_CACHE_DTYPES}, "
                f"got {self.kv_cache_dtype!r}"
            )
        if self.kv_quant_scheme not in ("int8",):
            raise ValueError(
                f"kv_quant_scheme must be 'int8' (per-token symmetric), "
                f"got {self.kv_quant_scheme!r}"
            )
        if self.mla_kv_latent_dim < 0:
            raise ValueError(
                f"mla_kv_latent_dim must be >= 0, got {self.mla_kv_latent_dim}"
            )
        if self.mla_rope_head_dim < 0 or self.mla_rope_head_dim % 2 != 0:
            raise ValueError(
                f"mla_rope_head_dim must be a non-negative even number (RoPE "
                f"operates on pairs), got {self.mla_rope_head_dim}"
            )
        _MOE_ENGINES = ("auto", "grouped", "per_expert")
        if self.moe_engine not in _MOE_ENGINES:
            raise ValueError(
                f"moe_engine must be one of {_MOE_ENGINES}, "
                f"got {self.moe_engine!r}"
            )
        if self.moe_group_ratio <= 0:
            raise ValueError(
                f"moe_group_ratio must be > 0, got {self.moe_group_ratio}"
            )
        if self.moe_cache_size < 0:
            raise ValueError(
                f"moe_cache_size must be >= 0, got {self.moe_cache_size}"
            )
        if self.moe_cache_bytes < 0:
            raise ValueError(
                f"moe_cache_bytes must be >= 0, got {self.moe_cache_bytes}"
            )
        _PACKING_STRATEGIES = ("stream", "bin")
        if self.packing_strategy not in _PACKING_STRATEGIES:
            raise ValueError(
                f"packing_strategy must be one of {_PACKING_STRATEGIES}, "
                f"got {self.packing_strategy!r}"
            )
        if self.use_packing and self.use_attention_sink:
            raise ValueError(
                "use_packing and use_attention_sink are incompatible: the "
                "attention sink prepends a token outside the packed layout. "
                "Disable one of them."
            )

        # Derived values
        self.head_dim = self.d_model // self.n_heads
        self.effective_batch_size = self.micro_batch_size * self.gradient_accumulation_steps

        # Create directories
        for d in [self.data_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ModelConfig":
        """Create config from a named preset with optional overrides."""
        if name not in PRESETS:
            available = ", ".join(PRESETS.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        preset = {k: v for k, v in PRESETS[name].items() if k != "description"}
        preset.update(overrides)
        return cls(**preset)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Filter out non-init fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values() if f.init}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def save_json(self, path: str) -> None:
        """Save configuration to a JSON file."""
        data = {}
        for k, v in asdict(self).items():
            if isinstance(v, Path):
                v = str(v)
            data[k] = v
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        """Return a human-readable, box-aligned configuration summary.

        The box width is fixed; every data row is padded so the closing ``│``
        always lines up regardless of label length.
        """
        WIDTH = 45  # inner width between the left and right box borders

        def row(content: str) -> str:
            """Pad ``content`` to WIDTH and wrap with box borders."""
            return f"│{content.ljust(WIDTH)}│"

        def kv(label: str, value) -> str:
            text = f"{label}: {value}"
            return row(f"    {text}")

        def title(text: str) -> str:
            return row(text.center(WIDTH))

        def section(text: str) -> str:
            return row(f"  {text}")

        lines = [
            "┌" + "─" * WIDTH + "┐",
            title("Μῆτις (Metis) Configuration"),
            "├" + "─" * WIDTH + "┤",
            section("Architecture"),
            kv("d_model", self.d_model),
            kv("n_heads", self.n_heads),
            kv("n_kv_heads", self.n_kv_heads),
            kv("n_groups", self.n_groups),
            kv("n_layers", self.n_layers),
            kv("vocab_size", self.vocab_size),
            kv("dropout", self.dropout),
            kv("RMSNorm", self.use_rmsnorm),
            kv("SwiGLU", self.use_swiglu),
            kv("RoPE", self.use_rope),
            kv("Attention", f"{self.attn_backend}" + ("" if self.use_flash_attn else " (math)")),
            "├" + "─" * WIDTH + "┤",
            section("Training"),
            kv("batch_size", self.effective_batch_size),
            kv("max_iters", self.max_iters),
            kv("learning_rate", self.learning_rate),
            kv("warmup_steps", self.warmup_steps),
            kv("grad_clip", self.max_grad_norm),
            "├" + "─" * WIDTH + "┤",
            kv("Device", self.device),
            kv("Seed", self.seed),
            kv("CUDA Graphs", self.use_cuda_graphs),
            kv("Overlap Pipeline", self.use_pipeline),
            kv("Prefetch Depth", self.prefetch_depth),
            kv("Async Checkpoint", self.async_checkpoint),
            kv("Exec Scheduler", self.use_exec_scheduler),
            kv("KV Backend", self._kv_summary()),
            kv("Parameters", self.n_params),
            "└" + "─" * WIDTH + "┘",
        ]
        return "\n".join(lines)

    def _kv_summary(self) -> str:
        """Human-readable KV subsystem description for the summary box."""
        if self.kv_backend == "mla":
            latent = self.mla_kv_latent_dim or (self.d_model // self.n_heads)
            rope = self.mla_rope_head_dim or (self.head_dim // 2)
            return f"MLA (latent={latent}, rope={rope})"
        if self.kv_backend == "quantized":
            return f"quantized ({self.kv_quant_scheme})"
        if self.kv_backend == "static":
            return f"static ({self.kv_cache_dtype})"
        return "default (growable)"
