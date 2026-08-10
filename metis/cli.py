"""
Μῆτις (Metis) — Unified Command-Line Interface
=================================================
Commands:

    metis train       Train a model from a text dataset
    metis distill     Train forever from an API teacher (distillation)
    metis generate    Generate text from a single prompt
    metis chat        Interactive streaming chat
    metis serve       Start a REST API server (FastAPI)
    metis ui          Launch a Gradio web UI in the browser
    metis info        Show model/checkpoint status at a glance
    metis find-lr     Learning rate range finder

Run ``metis --help`` or ``metis <command> --help`` for full options.
"""

import argparse
import logging
import os
import sys

from .config import PRESETS, ModelConfig, setup_logging
from .data import CharTokenizer, create_dataloader, load_text, train_val_split
from .generate import chat, generate_text, load_model_and_tokenizer
from .teacher import TeacherError
from .training import find_lr
from .training import train as run_training

logger = logging.getLogger("metis.cli")


def _configure_stdio(streams=None) -> None:
    """Make console output encoding-safe (Windows cp1252 vs UTF-8).

    The banner contains Greek and box-drawing characters that crash on a
    Windows console whose codepage can't represent them (UnicodeEncodeError).
    Reconfigure stdout/stderr to UTF-8 so printing is never terminal-dependent;
    ``errors="replace"`` is a last-resort guard for exotic forced encodings.
    """
    for stream in (streams or (sys.stdout, sys.stderr)):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # stream has no reconfigure (e.g. embedded); leave as-is


# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────

BANNER = r"""
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Μῆτις (Metis) v3.0 — modern tiny language model           │
│   RMSNorm · RoPE · SwiGLU · GQA · MoE · BPE · DDP          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
"""


def _error(msg: str) -> None:
    print(f"\n ❌ {msg}\n", file=sys.stderr)


def _apply_common_overrides(args) -> None:
    """Set env vars from --moe-cache-size / --moe-cache-bytes before model build."""
    if getattr(args, "moe_cache_size", None) is not None:
        os.environ["METIS_MOE_CACHE_SIZE"] = str(args.moe_cache_size)
    if getattr(args, "moe_cache_bytes", None) is not None:
        os.environ["METIS_MOE_CACHE_BYTES"] = str(args.moe_cache_bytes)


def _add_common(parser: argparse.ArgumentParser) -> None:
    """Options shared by commands that load a trained model."""
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu / cuda). Auto-detects if omitted.")
    parser.add_argument("--attn-backend", type=str, default=None,
                        choices=["auto", "flash_attn", "sdpa", "flash", "mem_efficient", "math"],
                        help="Attention backend override (default: saved config)")
    parser.add_argument("--kv-backend", type=str, default=None,
                        choices=["default", "static", "quantized", "mla"],
                        help="KV cache backend (default: saved config): "
                             "default = legacy growable, static = preallocated "
                             "buffers (bit-identical), quantized = int8 "
                             "compressed, mla = latent attention")
    parser.add_argument("--moe-cache-size", type=int, default=None,
                        help="Max entries in the persistent expert weight cache "
                             "(default from config; set 0 to disable)")
    parser.add_argument("--moe-cache-bytes", type=int, default=None,
                        help="Optional byte budget for the expert cache "
                             "(default from config; 0 = unbounded)")
    parser.add_argument("--log-level", type=str, default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: WARNING)")


# ── Parser builders ───────────────────────────────────────────────────────────

def _build_train_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "train",
        help="Train a model from a text dataset",
        description="Train a Μῆτις language model from a plain-text dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presets:\n"
            + "\n".join(f"  {k:8s}  {v['description']}" for k, v in PRESETS.items())
            + "\n\nExamples:\n"
            "  metis train --dataset data/input.txt\n"
            "  metis train --preset medium --iters 10000\n"
            "  metis train --resume\n"
            "  metis train --preset tiny --tokenizer cl100k_base --use_moe\n"
        ),
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--dataset", type=str, default="data/input.txt")
    p.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation steps; effective batch = "
                        "batch-size * grad-accum (default from preset)")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--optimizer", type=str, default=None,
                   choices=["adamw", "bnb8bit"],
                   help="Optimizer: adamw (default) | bnb8bit (bitsandbytes 8-bit Adam, "
                        "fits ~1B-param models in 16 GB VRAM)")
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-cuda-graphs", action="store_true",
                   help="Disable CUDA Graphs for the training step (default: on "
                        "when CUDA is available, with automatic eager fallback)")
    # Phase 6: Overlapped training pipeline
    p.add_argument("--no-pipeline", action="store_true",
                   help="Disable the overlapped pipeline (thread prefetch, async "
                        "H2D staging, async checkpoints) — the original serial path")
    p.add_argument("--prefetch-depth", type=int, default=None,
                   help="Steps read ahead by the prefetch thread (default: 2)")
    p.add_argument("--pipeline-buffer-depth", type=int, default=None,
                   help="H2D staging ring depth — copy-stream slack (default: 3)")
    p.add_argument("--no-async-checkpoint", action="store_true",
                   help="Write checkpoints synchronously on the main thread")
    p.add_argument("--no-layer-prefetch", action="store_true",
                   help="Disable layer prefetching (warm the next layer's expert "
                        "cache on a side stream during compute)")
    p.add_argument("--attn-backend", type=str, default=None,
                   choices=["auto", "flash_attn", "sdpa", "flash", "mem_efficient", "math"],
                   help="Attention backend (default: auto)")
    # Phase 1: Tokenizer
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer: char | cl100k_base | p50k_base | o200k_base (default: char)")
    # Phase 3: MoE
    p.add_argument("--use-moe", action="store_true", help="Enable Mixture of Experts")
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--moe-top-k", type=int, default=2)
    p.add_argument("--moe-engine", type=str, default=None,
                   choices=["auto", "grouped", "per_expert"],
                   help="MoE execution engine: grouped (token-sorted bmm) | "
                        "per_expert (legacy loop) | auto (default: grouped)")
    p.add_argument("--moe-group-ratio", type=float, default=None,
                   help="Max/min token ratio inside one expert group for the "
                        "grouped scheduler (default 2.0; lower = tighter blocks, "
                        "more bmm pairs)")
    p.add_argument("--moe-cache-size", type=int, default=None,
                   help="Max entries in the persistent expert weight cache "
                        "(default 64; set 0 to disable)")
    p.add_argument("--moe-cache-bytes", type=int, default=None,
                   help="Optional byte budget for the expert cache "
                        "(default 0 = unbounded by bytes)")
    # Phase 3: QK-Norm
    p.add_argument("--use-qk-norm", action="store_true", help="Enable QK-Normalization")
    # Phase 3: Attention Sink
    p.add_argument("--use-attention-sink", action="store_true",
                   help="Enable Attention Sink for extended context")
    # Phase 2: EMA
    p.add_argument("--use-ema", action="store_true", help="Enable EMA")
    # Phase 2: W&B
    p.add_argument("--use-wandb", action="store_true", help="Log to Weights & Biases")
    p.add_argument("--wandb-project", type=str, default="metis-llm")
    # Phase 1: Data
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    p.add_argument("--no-mmap", action="store_true", help="Disable memory-mapped dataset")
    p.add_argument("--force-recache", action="store_true",
                   help="Rebuild the tokenized data cache from scratch, ignoring "
                        "any existing cache file")
    # Phase 5: Dynamic Sequence Packing
    p.add_argument("--use-packing", action="store_true",
                   help="Enable dynamic sequence packing (packs short documents "
                        "into dense fixed-length batches — no padding waste)")
    p.add_argument("--packing-strategy", type=str, default=None,
                   choices=["stream", "bin"],
                   help="Packing strategy: stream = contiguous zero-pad (default) | "
                        "bin = whole-document first-fit-decreasing")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def _build_generate_parser(subparsers) -> None:
    p = subparsers.add_parser("generate",
        help="Generate text from a single prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--exec-scheduler", action="store_true",
                   help="Enable graph-based execution scheduler for decode")
    _add_common(p)


def _build_chat_parser(subparsers) -> None:
    p = subparsers.add_parser("chat",
        help="Interactive streaming chat",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--exec-scheduler", action="store_true",
                   help="Enable graph-based execution scheduler for decode")
    _add_common(p)


def _build_info_parser(subparsers) -> None:
    p = subparsers.add_parser("info",
        help="Show model & checkpoint status",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(p)


def _build_serve_parser(subparsers) -> None:
    p = subparsers.add_parser("serve",
        help="Start a REST API server (FastAPI)",
        description="Start a FastAPI-based REST API server for inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  metis serve --port 8000\n  metis serve --checkpoint-dir my_ckpt\n")
    p.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p.add_argument("--reload", action="store_true", help="Auto-reload")
    _add_common(p)


def _build_ui_parser(subparsers) -> None:
    p = subparsers.add_parser("ui",
        help="Launch Gradio web UI in browser",
        description="Launch a browser-based chat interface.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    p.add_argument("--share", action="store_true", help="Create public link")
    _add_common(p)


def _build_findlr_parser(subparsers) -> None:
    p = subparsers.add_parser("find-lr",
        help="Learning rate range finder",
        description="Run a learning rate range test to find the optimal peak LR.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, default="data/input.txt")
    p.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="tiny")
    p.add_argument("--start-lr", type=float, default=1e-7)
    p.add_argument("--end-lr", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])


def _build_distill_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "distill",
        help="Train forever from an API teacher (distillation)",
        description=(
            "Train Μῆτις continuously on text written by a frontier teacher "
            "model reached through an OpenAI-compatible API (your omniroute "
            "gateway). Runs until you stop it; restarting the same command "
            "resumes automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Teacher API (env vars, or --teacher-* flags):\n"
            "  METIS_TEACHER_BASE_URL   e.g. https://your-gateway.example/v1\n"
            "  METIS_TEACHER_API_KEY\n"
            "  METIS_TEACHER_MODEL      e.g. deepseek-chat\n"
            "  METIS_TEACHER_TIMEOUT    seconds per call (default 240)\n"
            "\n"
            "Stop anytime: Ctrl+C, or create a file named STOP in the "
            "checkpoint dir. Re-run the same command to resume - no setup.\n"
            "\n"
            "Examples:\n"
            "  metis distill --checkpoint-dir checkpoints_distill --preset tiny\n"
            "  metis distill --checkpoint-dir checkpoints_distill --test-teacher\n"
            "  metis distill --checkpoint-dir ckpt --mock --max-steps 10\n"
            "  metis distill --checkpoint-dir ckpt --topic animals --topic-file topics.txt\n"
        ),
    )
    _add_common(p)
    # Distillation runs are long-lived and users watch the log: default to INFO.
    p.set_defaults(log_level="INFO")
    p.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default=None,
                   help="Model-size preset (tiny/small/medium/large)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Micro-batch size (sequences per forward; default from preset)")
    p.add_argument("--seq-len", type=int, default=None,
                   help="Sequence length / context window (default from preset)")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation steps; effective batch = "
                        "batch-size * grad-accum (default from preset)")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer: cl100k_base (recommended, fixed vocab) | "
                        "char (fit once from --seed-data) | p50k_base | o200k_base "
                        "(default: cl100k_base)")
    p.add_argument("--seed-data", type=str, default=None,
                   help="Corpus file used to fit a char tokenizer once on first run")
    p.add_argument("--topic", type=str, default="general knowledge",
                   help="Topic the teacher should write about (default: general knowledge)")
    p.add_argument("--topic-file", type=str, default=None,
                   help="File with one topic per line - the teacher rotates through them")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after N optimizer steps (0 = run forever; default 0)")
    p.add_argument("--save-every", type=int, default=50,
                   help="Save checkpoint + state every N steps (default 50)")
    p.add_argument("--steps-per-call", type=int, default=4,
                   help="Optimizer steps to run per teacher API call (default 4)")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Max tokens per teacher call (default 1024)")
    p.add_argument("--min-sleep", type=float, default=1.0,
                   help="Minimum seconds between teacher calls - pace/cost guard "
                        "(default 1.0)")
    p.add_argument("--budget-tokens", type=int, default=0,
                   help="Stop after this many total teacher tokens "
                        "(0 = unlimited; default 0)")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing checkpoints/state and start fresh")
    p.add_argument("--mock", action="store_true",
                   help="Use a built-in mock teacher (offline test, no API)")
    p.add_argument("--test-teacher", action="store_true",
                   help="Make one connectivity call to the teacher API, print "
                        "the reply, then exit")
    p.add_argument("--teacher-base-url", type=str, default=None,
                   help="Overrides METIS_TEACHER_BASE_URL")
    p.add_argument("--teacher-api-key", type=str, default=None,
                   help="Overrides METIS_TEACHER_API_KEY")
    p.add_argument("--teacher-model", type=str, default=None,
                   help="Overrides METIS_TEACHER_MODEL")
    p.add_argument("--teacher-timeout", type=int, default=None,
                   help="Overrides METIS_TEACHER_TIMEOUT (seconds per call)")
    return p


# ── Command handlers ──────────────────────────────────────────────────────────

def _build_train_config(args) -> ModelConfig:
    if args.preset:
        config = ModelConfig.from_preset(args.preset, dataset_path=args.dataset)
    else:
        config = ModelConfig(dataset_path=args.dataset)

    config.checkpoint_dir = args.checkpoint_dir
    if args.iters is not None:
        config.max_iters = args.iters
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.batch_size is not None:
        config.micro_batch_size = args.batch_size
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.seq_len is not None:
        config.max_seq_len = args.seq_len
    if args.optimizer is not None:
        config.optimizer = args.optimizer
    if args.n_kv_heads is not None:
        config.n_kv_heads = args.n_kv_heads
    if args.tokenizer is not None:
        config.tokenizer = args.tokenizer
    config.seed = args.seed
    config.compile_model = args.compile
    config.use_cuda_graphs = not getattr(args, "no_cuda_graphs", False)
    config.log_level = args.log_level
    if args.attn_backend is not None:
        config.attn_backend = args.attn_backend
    config.use_moe = getattr(args, "use_moe", False)
    config.moe_num_experts = getattr(args, "num_experts", 8)
    config.moe_top_k = getattr(args, "moe_top_k", 2)
    config.moe_engine = getattr(args, "moe_engine", None) or config.moe_engine
    config.moe_group_ratio = getattr(args, "moe_group_ratio", None) or config.moe_group_ratio
    # Explicit CLI flags must win over a pre-existing METIS_MOE_CACHE_* env var
    # (MoE.__init__ reads the env var first, so mirror the flag into it).
    if getattr(args, "moe_cache_size", None) is not None:
        config.moe_cache_size = args.moe_cache_size
        os.environ["METIS_MOE_CACHE_SIZE"] = str(args.moe_cache_size)
    if getattr(args, "moe_cache_bytes", None) is not None:
        config.moe_cache_bytes = args.moe_cache_bytes
        os.environ["METIS_MOE_CACHE_BYTES"] = str(args.moe_cache_bytes)
    # Mirror env-derived cache settings into config so save_json / metis info
    # reflect the effective value (MoE.__init__ reads env before config).
    if getattr(args, "moe_cache_size", None) is None:
        env_val = os.environ.get("METIS_MOE_CACHE_SIZE")
        if env_val is not None:
            try:
                config.moe_cache_size = int(env_val)
            except ValueError:
                pass  # MoE.__init__ will handle the bad env value
    if getattr(args, "moe_cache_bytes", None) is None:
        env_val = os.environ.get("METIS_MOE_CACHE_BYTES")
        if env_val is not None:
            try:
                config.moe_cache_bytes = int(env_val)
            except ValueError:
                pass
    config.use_qk_norm = getattr(args, "use_qk_norm", False)
    config.use_attention_sink = getattr(args, "use_attention_sink", False)
    config.use_pipeline = not getattr(args, "no_pipeline", False)
    if getattr(args, "prefetch_depth", None) is not None:
        config.prefetch_depth = args.prefetch_depth
    if getattr(args, "pipeline_buffer_depth", None) is not None:
        config.pipeline_buffer_depth = args.pipeline_buffer_depth
    config.async_checkpoint = not getattr(args, "no_async_checkpoint", False)
    config.use_layer_prefetch = not getattr(args, "no_layer_prefetch", False)
    config.use_ema = getattr(args, "use_ema", False)
    config.use_wandb = getattr(args, "use_wandb", False)
    config.wandb_project = getattr(args, "wandb_project", "metis-llm")
    config.num_workers = getattr(args, "num_workers", 0)
    config.use_mmap = not getattr(args, "no_mmap", False)
    config.force_recache = getattr(args, "force_recache", False)
    config.use_packing = getattr(args, "use_packing", False)
    config.packing_strategy = getattr(args, "packing_strategy", None) or config.packing_strategy
    # Re-validate after CLI overrides: attribute assignment bypasses the
    # dataclass __post_init__ checks (e.g. an incompatible --n-kv-heads would
    # otherwise crash in the first forward pass), and refreshes the derived
    # head_dim / effective_batch_size fields.
    config.validate()
    return config


def cmd_train(args) -> int:
    config = _build_train_config(args)
    print(BANNER)
    try:
        run_training(config, resume=args.resume)
    except FileNotFoundError as e:
        _error(f"Training failed: {e}")
        return 1
    return 0


def _build_distill_config(args) -> ModelConfig:
    """Build the model config for distillation (preset/CLI + common overrides)."""
    if args.preset:
        config = ModelConfig.from_preset(args.preset, dataset_path="")
    else:
        config = ModelConfig(dataset_path="")
    config.checkpoint_dir = args.checkpoint_dir
    if args.tokenizer is not None:
        config.tokenizer = args.tokenizer
    if args.batch_size is not None:
        config.micro_batch_size = args.batch_size
    if args.seq_len is not None:
        config.max_seq_len = args.seq_len
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum
    if args.device:
        config.device = args.device
    if args.attn_backend is not None:
        config.attn_backend = args.attn_backend
    if args.kv_backend is not None:
        config.kv_backend = args.kv_backend
    config.log_level = args.log_level
    # The distill loop is self-contained: it does not use the overlapped
    # pipeline / CUDA graphs / packing machinery from train().
    config.use_pipeline = False
    config.use_cuda_graphs = False
    config.use_packing = False
    # Re-validate after CLI overrides (see _build_train_config).
    config.validate()
    return config


def _build_distill_options(args):
    from .distill import DistillOptions
    return DistillOptions(
        topic=args.topic,
        topic_file=args.topic_file,
        seed_data=args.seed_data,
        max_steps=args.max_steps,
        save_every=args.save_every,
        steps_per_call=args.steps_per_call,
        max_tokens=args.max_tokens,
        min_sleep=args.min_sleep,
        budget_tokens=args.budget_tokens,
        no_resume=args.no_resume,
        mock=args.mock,
        test_teacher=args.test_teacher,
        teacher_base_url=args.teacher_base_url,
        teacher_api_key=args.teacher_api_key,
        teacher_model=args.teacher_model,
        teacher_timeout=args.teacher_timeout,
    )


def cmd_distill(args) -> int:
    config = _build_distill_config(args)
    opts = _build_distill_options(args)
    print(BANNER)
    try:
        from .distill import distill
        return distill(config, opts)
    except (FileNotFoundError, ValueError) as e:
        _error(f"Distillation failed: {e}")
        return 1
    except TeacherError as e:
        _error(f"Teacher error: {e}")
        return 1


def cmd_generate(args) -> int:
    setup_logging(args.log_level)
    _apply_common_overrides(args)
    if getattr(args, "exec_scheduler", False):
        os.environ["METIS_EXEC_SCHEDULER"] = "1"
    try:
        model, tokenizer, config = load_model_and_tokenizer(
            args.checkpoint_dir, args.device, args.attn_backend,
            kv_backend=getattr(args, "kv_backend", None),
        )
    except (FileNotFoundError, SystemExit):
        _error(
            f"No checkpoint in '{args.checkpoint_dir}'. "
            "Train first: metis train --dataset data/input.txt"
        )
        return 1

    print(f"Prompt: {args.prompt}\n")

    def stream_token(token: str):
        print(token, end="", flush=True)

    generated = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=config.device,
        stream_callback=None if args.no_stream else stream_token,
        use_kv_cache=not args.no_cache,
    )

    if args.no_stream:
        print(generated)
    else:
        print()
    return 0


def cmd_chat(args) -> int:
    setup_logging(args.log_level)
    _apply_common_overrides(args)
    if getattr(args, "exec_scheduler", False):
        os.environ["METIS_EXEC_SCHEDULER"] = "1"
    try:
        model, tokenizer, config = load_model_and_tokenizer(
            args.checkpoint_dir, args.device, args.attn_backend,
            kv_backend=getattr(args, "kv_backend", None),
        )
    except (FileNotFoundError, SystemExit):
        _error(f"No checkpoint in '{args.checkpoint_dir}'. Train first.")
        return 1
    chat(model, tokenizer, config, stream=not args.no_stream)
    return 0


def cmd_info(args) -> int:
    setup_logging(args.log_level)
    ckpt_dir = args.checkpoint_dir
    print(BANNER)

    if not os.path.isdir(ckpt_dir):
        print(f"  Checkpoint directory: {os.path.abspath(ckpt_dir)}")
        print("    (not found — train a model first)")
        return 0

    print(f"  Checkpoint directory : {os.path.abspath(ckpt_dir)}\n")

    artifacts = [
        ("config.json", "Saved ModelConfig"),
        ("tokenizer.json", "Tokenizer (JSON)"),
        ("tokenizer.pkl", "Tokenizer (legacy pickle)"),
        ("best_model.pt", "Best-validation model"),
        ("final_model.pt", "Last-run model state"),
        ("latest_checkpoint.pt", "Resumable checkpoint"),
    ]
    found = 0
    for name, desc in artifacts:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            size_str = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size:,} bytes"
            print(f"    ✓ {name:<24} {size_str:<14} — {desc}")
            found += 1

    if found == 0:
        print("    (nothing here yet — train a model first)")
        print("\n    → metis train --dataset data/input.txt")
        return 0

    config_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(config_path):
        print("\n" + ModelConfig.from_json(config_path).summary())
    return 0


def cmd_serve(args) -> int:
    setup_logging(args.log_level)
    _apply_common_overrides(args)
    try:
        from .server import main as server_main
        # Forward args
        sys.argv = ["metis-serve",
                     "--host", args.host,
                     "--port", str(args.port),
                     "--checkpoint-dir", args.checkpoint_dir]
        if args.device:
            sys.argv += ["--device", args.device]
        if args.attn_backend is not None:
            os.environ["METIS_ATTN_BACKEND"] = args.attn_backend
        if getattr(args, "kv_backend", None) is not None:
            os.environ["METIS_KV_BACKEND"] = args.kv_backend
        if args.reload:
            sys.argv.append("--reload")
        server_main()
    except ImportError as e:
        _error(f"Cannot start server: {e}. Run: pip install fastapi uvicorn")
        return 1
    return 0


def cmd_ui(args) -> int:
    setup_logging(args.log_level)
    _apply_common_overrides(args)
    try:
        from .webui import main as webui_main
        sys.argv = ["metis-ui",
                     "--checkpoint-dir", args.checkpoint_dir,
                     "--port", str(args.port)]
        if args.device:
            sys.argv += ["--device", args.device]
        if args.share:
            sys.argv.append("--share")
        webui_main()
    except ImportError as e:
        _error(f"Cannot start UI: {e}. Run: pip install gradio")
        return 1
    return 0


def cmd_find_lr(args) -> int:
    """Run learning rate range finder."""
    setup_logging(args.log_level)
    print(BANNER)
    print("  Learning Rate Range Finder\n")


    # Build config
    config = ModelConfig.from_preset(args.preset, dataset_path=args.dataset)
    if args.seq_len is not None:
        config.max_seq_len = args.seq_len
    config.log_level = args.log_level

    # Load text
    try:
        text = load_text(config.dataset_path)
    except FileNotFoundError as e:
        _error(str(e))
        return 1

    # Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    config.vocab_size = tokenizer.vocab_size

    # Create loader
    train_text, _ = train_val_split(text, config.train_split)
    train_loader = create_dataloader(
        train_text, tokenizer, config.max_seq_len, config.micro_batch_size,
        shuffle=True,
    )

    # Create model
    from .model import MetisLM
    model = MetisLM(config)
    model.to(config.device)
    print(f"  Model: {config.n_params} parameters")
    print(f"  Running LR test from {args.start_lr:.0e} to {args.end_lr:.0e} "
          f"over {args.steps} steps...\n")

    lrs, losses = find_lr(model, train_loader, config,
                          start_lr=args.start_lr, end_lr=args.end_lr,
                          num_steps=args.steps)

    # Find the loss minimum point (suggested LR)
    min_idx = losses.index(min(losses))
    suggested_lr = lrs[min_idx]

    # Find steepest descent (best LR)
    gradients = [losses[i+1] - losses[i] for i in range(len(losses)-1)]
    steepest_idx = gradients.index(min(gradients))
    best_lr = lrs[steepest_idx]

    print("  Results:")
    print(f"    Suggested LR (loss minimum):   {suggested_lr:.2e}  "
          f"(loss={losses[min_idx]:.4f})")
    print(f"    Recommended LR (steepest drop): {best_lr:.2e}")
    print(f"    OneCycle start/end:            {best_lr / 10:.2e} / {best_lr:.2e}")
    print()

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("LR Range Test — Μῆτις (Metis)")
        plt.axvline(x=best_lr, color="r", linestyle="--", alpha=0.5,
                    label=f"Recommended: {best_lr:.2e}")
        plt.axvline(x=suggested_lr, color="g", linestyle="--", alpha=0.5,
                    label=f"Min loss: {suggested_lr:.2e}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("lr_finder.png")
        print("  Plot saved: lr_finder.png")
    except ImportError:
        print("  (install matplotlib to generate a plot)")

    return 0


# ── Dispatch ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metis",
        description="Μῆτις (Metis) v3.0 — modern tiny language model, built from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n"
            "  train      Train a model from a text dataset\n"
            "  distill    Train forever from an API teacher (distillation)\n"
            "  generate   Generate text from a single prompt\n"
            "  chat       Interactive streaming chat\n"
            "  serve      Start a REST API server (FastAPI)\n"
            "  ui         Launch Gradio web UI in browser\n"
            "  info       Show model & checkpoint status\n"
            "  find-lr    Learning rate range finder\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _build_train_parser(subparsers)
    _build_distill_parser(subparsers)
    _build_generate_parser(subparsers)
    _build_chat_parser(subparsers)
    _build_info_parser(subparsers)
    _build_serve_parser(subparsers)
    _build_ui_parser(subparsers)
    _build_findlr_parser(subparsers)
    return parser


def main(argv=None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "train": cmd_train,
        "distill": cmd_distill,
        "generate": cmd_generate,
        "chat": cmd_chat,
        "info": cmd_info,
        "serve": cmd_serve,
        "ui": cmd_ui,
        "find-lr": cmd_find_lr,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
