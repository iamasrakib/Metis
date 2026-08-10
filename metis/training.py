"""
Μῆτις (Metis) — Advanced Training Pipeline
=============================================
Professional training with:
  • Progress bars (tqdm)
  • Train/validation loss tracking
  • Gradient clipping & accumulation
  • Cosine LR schedule with warmup
  • Automatic Mixed Precision (AMP)
  • Checkpointing with resume support
  • Periodic sample generation
  • Distributed Data Parallel (DDP)
  • Exponential Moving Average (EMA)
  • Weights & Biases experiment tracking
  • Memory-mapped datasets for GB-scale data
  • BPE tokenizer support
  • Reproducible seeding
"""

import gc
import logging
import math
import os
import time

import torch
import torch.nn as nn

from .config import ModelConfig, get_amp_dtype, setup_logging
from .cuda_graphs import CUDAGraphStep
from .data import (
    BPETokenizer,
    CharTokenizer,
    create_dataloader,
    create_packed_dataloader,
    load_documents,
    load_text,
    split_documents,
    train_val_split,
)
from .generate import generate_text
from .model import MetisLM
from .packing import PackedBatch
from .pipeline import (
    AsyncCheckpointer,
    GpuBatchStager,
    GpuIdleTracker,
    ThreadPrefetcher,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger("metis.train")


# ──────────────────────────────────────────────────────────────────────────────
# Learning Rate Schedule
# ──────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, config: ModelConfig) -> float:
    """Cosine decay learning rate with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= config.max_iters:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps) / (config.max_iters - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def find_lr(
    model: MetisLM,
    train_loader,
    config: ModelConfig,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    beta: float = 0.98,
    num_steps: int = 100,
) -> tuple[list, list]:
    """Learning rate range finder (Cyclical LR testing).

    Returns (lrs, losses) for plotting. The optimal LR is near the
    steepest descent point before divergence.
    """
    optimizer = model.configure_optimizers(
        config.weight_decay, start_lr, config.device, optimizer=config.optimizer
    )
    model.train()

    lrs, losses = [], []
    best_loss = float("inf")
    avg_loss = 0.0
    log_lr = math.log(start_lr)
    log_lr_end = math.log(end_lr)

    data_iter = iter(train_loader)
    for step in range(num_steps):
        lr = math.exp(log_lr + (log_lr_end - log_lr) * step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x, y, extra = _split_batch(batch)
        x, y = x.to(config.device), y.to(config.device)
        extra = {k: v.to(config.device) for k, v in extra.items()}
        optimizer.zero_grad(set_to_none=True)
        logits, loss, _ = model(x, y, **extra)
        loss.backward()
        optimizer.step()
        model.invalidate_moe_caches()

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed = avg_loss / (1 - beta ** (step + 1))
        lrs.append(lr)
        losses.append(smoothed)

        if step > 0 and smoothed > best_loss * 4:
            break
        if smoothed < best_loss:
            best_loss = smoothed

    return lrs, losses


# ──────────────────────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of all parameters, updated as:
        shadow = decay * shadow + (1 - decay) * param

    At validation/test time, EMA weights often generalize better
    than the live weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register(model)

    def _register(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Pre-allocate once: ``update`` mutates this buffer in place so
                # its storage address never changes (required under CUDA graphs
                # and for zero-allocation training steps).
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                # shadow <- decay * shadow + (1 - decay) * param, in place.
                # (The original ``new_average.clone()`` reallocated every step
                # and could leave a CUDA-graph-captured reference stale.)
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Save current params and replace with shadow (for validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                # ``copy_`` (not ``param.data = ...``) keeps the parameter's
                # storage address stable so a captured CUDA graph still writes
                # to the live weights after restore.
                param.data.copy_(self.shadow[name])
        # Invalidate MoE expert caches so stale entries from the live weights
        # are not served against the shadow weights (the framework contract).
        getattr(model, "invalidate_moe_caches", lambda: None)()

    def restore(self, model: nn.Module) -> None:
        """Restore params from backup after validation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
        # Invalidate MoE expert caches so stale entries from the shadow
        # weights are not served against the restored live weights.
        getattr(model, "invalidate_moe_caches", lambda: None)()


# ──────────────────────────────────────────────────────────────────────────────
# DDP Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _split_batch(batch):
    """Normalise a training batch into ``(input_ids, labels, forward_kwargs)``.

    Accepts both the classic ``(x, y)`` pair (yields empty kwargs) and a
    :class:`PackedBatch` (yields its block-diagonal ``attention_mask`` and
    per-segment ``position_ids`` as forward kwargs).
    """
    if isinstance(batch, PackedBatch):
        return batch.input_ids, batch.labels, batch.model_kwargs
    x, y = batch
    return x, y, {}


def _fetch_micro_batches(data_iter, train_loader, n: int):
    """Fetch ``n`` micro-batches, restarting the loader on exhaustion.

    Returns ``(batches, data_iter)`` so a restarted loader is reflected back
    to the caller — identical consumption semantics to the inline loop.
    """
    batches = []
    for _ in range(n):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        batches.append((x, y))
    return batches, data_iter


def is_main_process() -> bool:
    """Check if current process is the main DDP process."""
    return not torch.distributed.is_available() or not torch.distributed.is_initialized() \
        or torch.distributed.get_rank() == 0


def cleanup_ddp() -> None:
    """Clean up DDP process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(
    model: MetisLM,
    train_loader,
    val_loader,
    config: ModelConfig,
    ema: EMA | None = None,
) -> dict:
    """Estimate average loss and perplexity on train and validation sets."""
    model.eval()
    losses = {}
    use_amp = config.device.startswith("cuda")
    amp_dtype = get_amp_dtype(config.device)

    # Use EMA weights for validation if available
    if ema is not None:
        ema.apply_shadow(model)

    for split, loader in [("train", train_loader), ("val", val_loader)]:
        if loader is None:
            continue
        total_loss = 0.0
        data_iter = iter(loader)
        for _ in range(config.val_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            x, y, extra = _split_batch(batch)
            nb = config.device.startswith("cuda")
            x, y = x.to(config.device, non_blocking=nb), y.to(config.device, non_blocking=nb)
            extra = {k: v.to(config.device, non_blocking=nb) for k, v in extra.items()}
            with torch.autocast(device_type=config.device, dtype=amp_dtype, enabled=use_amp):
                _, loss, _ = model(x, y, **extra)
            total_loss += loss.item()
        avg_loss = total_loss / config.val_steps
        losses[split] = avg_loss
        losses[f"{split}_ppl"] = math.exp(avg_loss)

    # Restore live weights
    if ema is not None:
        ema.restore(model)

    model.train()
    return losses


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint Management
# ──────────────────────────────────────────────────────────────────────────────

def build_checkpoint(
    model: MetisLM,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    step: int,
    best_val_loss: float,
    ema: EMA | None = None,
) -> dict:
    """Assemble the checkpoint dict (state_dict D2H is synchronous, O(model))."""
    # Clone to CPU synchronously: state_dict() returns live GPU tensor
    # references that fused AdamW mutates in place; the async writer thread
    # must not race with the main thread's optimizer.step().
    model_sd = {
        k: v.detach().cpu() for k, v in model.state_dict().items()
    }
    checkpoint = {
        "model_state_dict": model_sd,
        "optimizer_state_dict": {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer.state_dict().items()
        },
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        "step": step,
        "best_val_loss": best_val_loss,
        "tokenizer_type": config.tokenizer,
        "version": "3.0",
    }
    if ema is not None:
        checkpoint["ema_shadow"] = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in ema.shadow.items()
        }
    return checkpoint


def build_checkpoint_raw(
    model: MetisLM,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    step: int,
    best_val_loss: float,
    ema: EMA | None = None,
) -> dict:
    """Assemble the checkpoint dict with **device tensors** left in place.

    The caller (``AsyncCheckpointer.submit_async``) issues the D2H on a copy
    stream so it overlaps the next step, instead of the synchronous
    ``detach().cpu()`` clone in :func:`build_checkpoint`.
    """
    checkpoint = {
        "model_state_dict": {k: v.detach() for k, v in model.state_dict().items()},
        "optimizer_state_dict": {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in optimizer.state_dict().items()
        },
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        "step": step,
        "best_val_loss": best_val_loss,
        "tokenizer_type": config.tokenizer,
        "version": "3.0",
    }
    if ema is not None:
        checkpoint["ema_shadow"] = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in ema.shadow.items()
        }
    return checkpoint


def save_checkpoint(
    model: MetisLM,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    step: int,
    best_val_loss: float,
    path: str,
    ema: EMA | None = None,
) -> None:
    """Save a training checkpoint (synchronous)."""
    checkpoint = build_checkpoint(model, optimizer, config, step, best_val_loss, ema)
    torch.save(checkpoint, path)
    if is_main_process():
        logger.info(f"Checkpoint saved → {path} (step {step})")


def _optimizer_compatible(optimizer, saved: dict) -> bool:
    """True if a saved optimizer state maps onto the current optimizer layout.

    The fused block merges q/k/v into one ``qkv`` parameter (and w1/w3 into
    ``w13``), so a checkpoint written before the fusion has an optimizer state
    keyed to the old, larger parameter set. ``Optimizer.load_state_dict`` would
    either raise on the group-size mismatch or silently mis-map state onto the
    wrong parameters — neither is acceptable on resume, so we detect the layout
    mismatch up front and fall back to a fresh optimizer (model weights still
    load via the state-dict compat shim).
    """
    saved_groups = saved.get("param_groups")
    if saved_groups is None or len(optimizer.param_groups) != len(saved_groups):
        return False
    current = [p for g in optimizer.param_groups for p in g["params"]]
    saved_ids = [i for g in saved_groups for i in g["params"]]
    if len(current) != len(saved_ids):
        return False
    saved_state = saved.get("state", {})
    for pos, (param, saved_id) in enumerate(zip(current, saved_ids)):
        for key, tensor in saved_state.get(str(saved_id), {}).items():
            if key == "step":
                continue  # scalar step counter, not a param-shaped buffer
            if tuple(tensor.shape) != tuple(param.shape):
                return False
    return True


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix ``torch.compile`` adds to state keys.

    A compile-wrapped model saves keys like ``_orig_mod.tok_emb.weight``; a
    checkpoint written from such a model (e.g. an older ``--compile-model``
    run, or one that saved the wrapped object instead of the unwrapped
    ``raw_model``) must still load into a fresh, uncompiled ``MetisLM``, whose
    ``load_state_dict`` would otherwise reject every key.
    """
    if any(k.startswith("_orig_mod.") for k in state_dict):
        return {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
    path: str,
    model: MetisLM,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Load a training checkpoint and return state info."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(_strip_compile_prefix(ckpt["model_state_dict"]))
    if "optimizer_state_dict" in ckpt:
        saved = ckpt["optimizer_state_dict"]
        if _optimizer_compatible(optimizer, saved):
            optimizer.load_state_dict(saved)
        else:
            logger.warning(
                "Optimizer state layout is from the pre-fusion parameter set "
                "(q/k/v and w1/w3 are now fused) — resuming with a fresh "
                "optimizer; model weights loaded normally."
            )
    step = ckpt.get("step", 0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    logger.info(
        f"Checkpoint loaded ← {path} "
        f"(step {step}, best_val_loss={best_val_loss:.4f})"
    )
    return {"step": step, "best_val_loss": best_val_loss, "ckpt": ckpt}


# ──────────────────────────────────────────────────────────────────────────────
# W&B Tracking
# ──────────────────────────────────────────────────────────────────────────────

def _init_wandb(config: ModelConfig) -> None:
    """Initialize Weights & Biases run if enabled."""
    if not config.use_wandb:
        return
    try:
        import wandb
        run_name = config.wandb_run_name or (
            f"metis-{config.tokenizer}-{config.d_model}d-{config.n_layers}l"
        )
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        )
        logger.info(f"W&B initialized: {config.wandb_project}/{run_name}")
    except (ImportError, Exception) as e:
        logger.warning(f"W&B initialization failed: {e}")


def _log_wandb(metrics: dict, step: int) -> None:
    """Log metrics to W&B if active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except (ImportError, Exception):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(config: ModelConfig, resume: bool = False) -> None:
    """Run the training loop given a fully-built ``ModelConfig``.

    Args:
        config: A configured ModelConfig.
        resume: If True, continue from latest checkpoint in ``config.checkpoint_dir``.
    """
    # ── Distributed setup ────────────────────────────────────────────────
    ddp_rank = int(os.environ.get("RANK", 0))
    ddp_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))

    if ddp_world_size > 1 and config.use_ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(ddp_local_rank)
        config.device = f"cuda:{ddp_local_rank}"
        logger.info(f"DDP initialized: rank {ddp_rank}/{ddp_world_size}, device {config.device}")
    else:
        config.use_ddp = False

    # ── Setup ─────────────────────────────────────────────────────────────
    if is_main_process():
        setup_logging(config.log_level, config.log_dir)
        logger.info("=" * 60)
        logger.info("  Μῆτις (Metis) v3.0 — Advanced Training Pipeline")
        logger.info("=" * 60)
        logger.info(f"  Tokenizer: {config.tokenizer}")
        logger.info(f"  Device:    {config.device}")
        ddp_status = f"enabled ({ddp_world_size} GPUs)" if config.use_ddp else "disabled"
        logger.info(f"  DDP:       {ddp_status}")

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(config.seed + ddp_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + ddp_rank)
    # deterministic=True disables cuDNN's auto-tuner so results are
    # reproducible across runs on the same hardware/driver.  Removing
    # benchmark=True (which contradicts deterministic by searching for the
    # fastest non-deterministic algorithm) keeps the determinism guarantee.
    # For max performance, users should set compile_model=True instead.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ── Prepare checkpoint directory ──────────────────────────────────────
    # The tokenizer + model checkpoints are written here; create it up front
    # so a fresh run never fails when the directory doesn't exist yet.
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    try:
        text = load_text(config.dataset_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("  Provide a dataset: metis train --dataset data/input.txt")
        raise

    # ── Tokenizer ─────────────────────────────────────────────────────────
    if config.tokenizer == "char":
        tokenizer = CharTokenizer()
    else:
        tokenizer = BPETokenizer(encoding_name=config.tokenizer)

    tokenizer.fit(text)
    tokenizer_path = os.path.join(config.checkpoint_dir, "tokenizer.json")
    if is_main_process():
        tokenizer.save(tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    config.pad_id = tokenizer.pad_id

    # ── Data Splits ───────────────────────────────────────────────────────
    train_text, val_text = train_val_split(text, config.train_split)

    # Adjust batch size for DDP (global batch = micro_batch * world_size)
    micro_batch = config.micro_batch_size
    if config.use_ddp:
        micro_batch = max(1, micro_batch // ddp_world_size)

    if config.use_packing:
        # Dynamic sequence packing: documents are tokenized individually and
        # packed into fixed-length sequences (metis/packing.py). Packed batches
        # carry a block-diagonal causal attention mask and per-segment RoPE
        # positions, so every slot in the batch is a real token.
        documents = load_documents(config.dataset_path)
        train_docs, val_docs = split_documents(documents, config.train_split)
        if config.use_ddp:
            # Shard the document lists per rank (PackedDataset is iterable, so
            # DistributedSampler does not apply) — each rank packs its own
            # disjoint slice instead of duplicating the whole corpus.
            train_docs = train_docs[ddp_rank::ddp_world_size]
            val_docs = val_docs[ddp_rank::ddp_world_size]
        if is_main_process():
            logger.info(
                f"Dynamic packing enabled ({config.packing_strategy} strategy): "
                f"{len(train_docs):,} train / {len(val_docs):,} val documents"
            )
        train_loader = create_packed_dataloader(
            train_docs, tokenizer, config.max_seq_len, micro_batch,
            strategy=config.packing_strategy, shuffle=True, seed=config.seed,
        )
        val_loader = create_packed_dataloader(
            val_docs, tokenizer, config.max_seq_len, micro_batch,
            strategy=config.packing_strategy, shuffle=False, seed=config.seed,
        )
    else:
        train_loader = create_dataloader(
            train_text, tokenizer, config.max_seq_len, micro_batch,
            shuffle=True, use_mmap=config.use_mmap,
            num_workers=config.num_workers,
            dataset_path=config.dataset_path,
            force_recache=config.force_recache,
            rank=ddp_rank, world_size=ddp_world_size,
        )
        val_loader = create_dataloader(
            val_text, tokenizer, config.max_seq_len, micro_batch,
            shuffle=False, use_mmap=config.use_mmap,
            num_workers=config.num_workers,
            dataset_path=config.dataset_path,
            rank=ddp_rank, world_size=ddp_world_size,
        )

    # The raw corpus strings were only needed to tokenize — the loaders now
    # hold compact token arrays. Free them before building the model: a 2 GB
    # corpus kept alive as text + slices (~4 GB) plus the fp32 model pushes a
    # Colab free-tier T4 runtime over its ~12 GB RAM budget.
    del text, train_text, val_text
    gc.collect()

    # ── Model ─────────────────────────────────────────────────────────────
    model = MetisLM(config)
    model.to(config.device)
    if is_main_process():
        logger.info(f"Model created — {config.n_params} parameters")
        try:
            _be = model.get_attention_backend()
            logger.info(
                f"Attention backend: requested={_be['requested']}, "
                f"machine-recommended={_be['recommended']}"
            )
        except Exception:
            pass
        print(config.summary())

    # ── DDP Wrapper ───────────────────────────────────────────────────────
    if config.use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp_local_rank], output_device=ddp_local_rank
        )

    # ── torch.compile ─────────────────────────────────────────────────────
    if config.compile_model and hasattr(torch, "compile"):
        if is_main_process():
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────────
    # Unwrap every wrapper (torch.compile → DDP → raw). Optimizer params and
    # checkpoint state_dict keys must come from the TRUE module — iterating the
    # compiled wrapper can emit "_orig_mod."-prefixed keys that a later
    # load_state_dict into an uncompiled model would reject.
    raw_model = model
    if hasattr(raw_model, "_orig_mod"):              # torch.compile wrapper
        raw_model = raw_model._orig_mod
    if config.use_ddp and hasattr(raw_model, "module"):
        raw_model = raw_model.module
    optimizer = raw_model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        config.device,
        optimizer=config.optimizer,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(config.device.startswith("cuda")))

    # ── EMA ───────────────────────────────────────────────────────────────
    ema = None
    if config.use_ema and not config.use_ddp:
        ema = EMA(raw_model, decay=config.ema_decay)
        if is_main_process():
            logger.info(f"EMA enabled (decay={config.ema_decay})")

    # ── Resume ────────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    checkpoint_dir = config.checkpoint_dir

    if resume:
        ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(ckpt_path):
            state = load_checkpoint(ckpt_path, raw_model, optimizer, config.device)
            start_step = state["step"] + 1
            best_val_loss = state["best_val_loss"]
            # Restore EMA shadow if present
            if ema is not None and "ema_shadow" in state.get("ckpt", {}):
                ema.shadow = state["ckpt"]["ema_shadow"]
        elif is_main_process():
            logger.warning(f"No checkpoint at {ckpt_path} — starting from scratch")

    # ── Save config ───────────────────────────────────────────────────────
    if is_main_process():
        config.save_json(os.path.join(checkpoint_dir, "config.json"))
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    has_best_model = os.path.exists(best_model_path)

    # ── W&B ───────────────────────────────────────────────────────────────
    if is_main_process():
        _init_wandb(config)

    # ── Training Loop ─────────────────────────────────────────────────────
    raw_model.train()
    data_iter = iter(train_loader)
    train_losses = []
    t0 = time.time()

    # ── Overlapped pipeline ───────────────────────────────────────────────
    # Software-pipelines the step: a background thread prefetches steps from
    # the loader (hiding disk/tokenization/preprocessing), a copy stream
    # stages the next micro-batch's H2D while the current one computes, and
    # checkpoints are written on a background thread. Opt-out via
    # ``--no-pipeline`` preserves the original serial path bit-for-bit.
    use_pipeline = config.use_pipeline
    prefetcher = None
    stager = None
    idle_tracker = None
    checkpointer = None
    cur_batches = None
    if use_pipeline:
        prefetcher = ThreadPrefetcher(
            train_loader,
            micro_batches=config.gradient_accumulation_steps,
            prefetch_depth=config.prefetch_depth,
            pin=config.device.startswith("cuda"),
        ).start()
        stager = GpuBatchStager(config.device, depth=config.pipeline_buffer_depth)
        idle_tracker = GpuIdleTracker(config.device, enabled=True)
        if config.async_checkpoint:
            checkpointer = AsyncCheckpointer(max_pending=1)
        if is_main_process():
            logger.info(
                "Overlapped pipeline: prefetch_depth=%d buffer_depth=%d "
                "async_checkpoint=%s",
                config.prefetch_depth, config.pipeline_buffer_depth,
                config.async_checkpoint,
            )

    use_amp = config.device.startswith("cuda")
    amp_dtype = get_amp_dtype(config.device)

    # ── CUDA Graphs ──────────────────────────────────────────────────────
    # Captures the whole gradient-accumulation iteration (N × fwd+bwd) as one
    # graph and replays it per step. Falls back to the eager loop when the
    # platform/model can't be captured. When active, the training step runs
    # non-checkpointed (capture is incompatible with checkpoint RNG); the
    # eager fallback keeps checkpointing, matching the original loop.
    cuda_graphs = None
    if config.use_cuda_graphs:
        if config.use_packing:
            # Packed batches carry data-shaped masks (block-diagonal per packed
            # sequence); capturing them into a static graph is possible but
            # brittle, so packing runs the eager training loop.
            if is_main_process():
                logger.info(
                    "CUDA Graphs: disabled — dynamic packing runs the eager loop"
                )
        else:
            cuda_graphs = CUDAGraphStep(
                raw_model, optimizer, scaler, config,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                micro_batch_size=micro_batch,
                max_seq_len=config.max_seq_len,
                amp_dtype=amp_dtype,
                device=config.device,
            )
            if is_main_process():
                _cg_info = cuda_graphs.info()
                _status = "active (graph replay)" if _cg_info["active"] else \
                    f"inactive — eager ({_cg_info['reason']})"
                logger.info(f"CUDA Graphs: {_status}")

    # Progress bar
    step_range = range(start_step, config.max_iters)
    if tqdm is not None and is_main_process():
        pbar = tqdm(
            step_range,
            desc="Training",
            total=config.max_iters - start_step,
            unit="step",
            bar_format="{l_bar}{bar:30}{r_bar}",
            dynamic_ncols=True,
        )
    else:
        pbar = step_range

    # Seed the prefetched pipeline: pull the first step's batches and, for the
    # CUDA-graph path, stage them into the static slots on the copy stream.
    if use_pipeline and prefetcher is not None:
        cur_batches = prefetcher.next_step()
        if cuda_graphs is not None and cuda_graphs.active:
            cuda_graphs.stage_next(cur_batches)
        elif stager is not None:
            # Eager path: issue the first micro-batch's H2D now so step 0
            # overlaps its transfer with the first forward.
            stager.stage(cur_batches[0])

    try:
        for step in pbar:
                # Update learning rate
                lr = get_lr(step, config)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                if idle_tracker is not None:
                    idle_tracker.begin()

                if cuda_graphs is not None:
                    # CUDA-graph backed step (or its eager fallback). Owns zero_grad,
                    # the micro-batch loop, gradient scaling and clipping.
                    if use_pipeline:
                        nxt = prefetcher.next_step()
                        # Replay the staged cur_batches; stage nxt on the copy stream
                        # during this replay so the next step's H2D overlaps compute.
                        loss_accum, grad_norm = cuda_graphs.train_step(
                            cur_batches, prefetch_next=nxt
                        )
                        cur_batches = nxt
                    else:
                        batches, data_iter = _fetch_micro_batches(
                            data_iter, train_loader, config.gradient_accumulation_steps
                        )
                        loss_accum, grad_norm = cuda_graphs.train_step(batches)
                else:
                    # Original eager step (gradient checkpointing on).
                    optimizer.zero_grad(set_to_none=True)
                    loss_accum = 0.0

                    if use_pipeline:
                        # Overlapped micro-batch loop: stage the next micro-batch on
                        # the copy stream *before* computing the current one, so each
                        # H2D transfer overlaps the previous forward/backward. The
                        # first micro-batch of this step was staged during the
                        # previous step's optimizer/EMA tail (cross-step overlap).
                        step_batches = cur_batches if cur_batches is not None else \
                            prefetcher.next_step()
                        cur_batches = None
                        for micro_step in range(config.gradient_accumulation_steps):
                            if micro_step + 1 < config.gradient_accumulation_steps:
                                stager.stage(step_batches[micro_step + 1])
                            x, y, extra = stager.device()
                            with torch.autocast(
                                device_type=config.device.split(":")[0], dtype=amp_dtype,
                                enabled=use_amp,
                            ):
                                use_grad_ckpt = config.device.startswith("cuda")
                                # Forward through the FULL wrapper stack (`model`):
                                # the compiled and/or DDP wrapper is a silent no-op
                                # if the loop calls the unwrapped `raw_model`.
                                logits, loss, _ = model(
                                    x, y, use_checkpointing=use_grad_ckpt, **extra
                                )
                                loss = loss / config.gradient_accumulation_steps
                            scaler.scale(loss).backward()
                            loss_accum += loss.item()
                            stager.mark_done()
                        # Cross-step overlap: pull the next step and stage its first
                        # micro-batch now, so that H2D runs during the optimizer step /
                        # EMA / logging tail below instead of stalling the next step.
                        if idle_tracker is not None:
                            idle_tracker.tick("compute")
                        cur_batches = prefetcher.next_step()
                        if idle_tracker is not None:
                            idle_tracker.tick("data_wait")
                        stager.stage(cur_batches[0])
                        if idle_tracker is not None:
                            idle_tracker.tick("h2d")
                    else:
                        for micro_step in range(config.gradient_accumulation_steps):
                            try:
                                batch = next(data_iter)
                            except StopIteration:
                                data_iter = iter(train_loader)
                                batch = next(data_iter)

                            x, y, extra = _split_batch(batch)
                            x, y = x.to(config.device), y.to(config.device)
                            extra = {k: v.to(config.device) for k, v in extra.items()}

                            with torch.autocast(
                                device_type=config.device.split(":")[0], dtype=amp_dtype,
                                enabled=use_amp,
                            ):
                                use_grad_ckpt = config.device.startswith("cuda")
                                # Forward through the FULL wrapper stack (`model`)
                                # so torch.compile / DDP wrappers actually run.
                                logits, loss, _ = model(
                                    x, y, use_checkpointing=use_grad_ckpt, **extra
                                )
                                loss = loss / config.gradient_accumulation_steps

                            scaler.scale(loss).backward()
                            loss_accum += loss.item()

                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(), config.max_grad_norm
                    )

                    # Ensure any in-flight async checkpoint snapshot has finished its
                    # D2H before this step's optimizer mutates the weights in place
                    # (no torn checkpoints). No-op between save steps.
                    if checkpointer is not None:
                        checkpointer.wait_pending()

                    if idle_tracker is not None:
                        idle_tracker.tick("optimizer")

                    scaler.step(optimizer)
                    scaler.update()

                # Invalidate MoE expert caches after weight update (fused AdamW
                # mutates weights without bumping _version — explicit invalidation
                # required for cache correctness).
                raw_model.invalidate_moe_caches()

                # EMA update
                if ema is not None:
                    ema.update(raw_model)

                train_losses.append(loss_accum)

                # GPU idle measurement (CUDA): wall vs busy time for this step.
                idle_row = idle_tracker.end() if idle_tracker is not None else None

                # ── Logging ───────────────────────────────────────────────────
                if step % config.log_interval == 0 and is_main_process():
                    elapsed = time.time() - t0
                    steps_per_sec = (step - start_step + 1) / elapsed if elapsed > 0 else 0
                    idle_str = f" │ idle {idle_row['idle_pct']:.1f}%" if idle_row else ""
                    if tqdm is not None:
                        pbar.set_postfix(
                            loss=f"{loss_accum:.4f}",
                            lr=f"{lr:.2e}",
                            grad=f"{grad_norm:.2f}",
                            **({"idle": f"{idle_row['idle_pct']:.1f}%"} if idle_row else {}),
                        )
                    else:
                        logger.info(
                            f"step {step:>6d}/{config.max_iters} │ "
                            f"loss {loss_accum:.4f} │ lr {lr:.2e} │ "
                            f"grad_norm {grad_norm:.2f} │ "
                            f"{steps_per_sec:.1f} steps/s"
                            f"{idle_str}"
                        )
                    wandb_metrics = {
                        "train/loss": loss_accum,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/step_per_sec": steps_per_sec,
                    }
                    if idle_row:
                        wandb_metrics["train/gpu_idle_pct"] = idle_row["idle_pct"]
                    _log_wandb(wandb_metrics, step)

                # ── Validation ────────────────────────────────────────────────
                if step > 0 and step % config.val_interval == 0 and is_main_process():
                    # Use raw_model for validation
                    val_model = raw_model
                    losses = estimate_loss(val_model, train_loader, val_loader, config, ema=ema)
                    logger.info(
                        f"  ╰─ Validation │ "
                        f"train_loss={losses.get('train', float('nan')):.4f} "
                        f"(ppl={losses.get('train_ppl', float('nan')):.2f}) │ "
                        f"val_loss={losses.get('val', float('nan')):.4f} "
                        f"(ppl={losses.get('val_ppl', float('nan')):.2f})"
                    )
                    _log_wandb({
                        "val/train_loss": losses.get("train", 0),
                        "val/val_loss": losses.get("val", 0),
                        "val/train_ppl": losses.get("train_ppl", 0),
                        "val/val_ppl": losses.get("val_ppl", 0),
                    }, step)

                    # Save best model
                    if "val" in losses and losses["val"] < best_val_loss:
                        best_val_loss = losses["val"]
                        torch.save(val_model.state_dict(), best_model_path)
                        has_best_model = True
                        logger.info(f"  ╰─ New best model saved (val_loss={best_val_loss:.4f})")

                # ── Sample Generation ─────────────────────────────────────────
                if step > 0 and step % config.sample_interval == 0 and is_main_process():
                    val_model = raw_model
                    val_model.eval()
                    prompt = "User: Hello\nMetis:"
                    sample = generate_text(
                        val_model, tokenizer, prompt,
                        max_new_tokens=80, temperature=0.8,
                        top_k=40, top_p=0.9, device=config.device,
                    )
                    response = sample[len(prompt):].strip().split("\n")[0]
                    logger.info(f"  ╰─ Sample │ \"{prompt}\" → \"{response}\"")
                    _log_wandb({"sample": response}, step)
                    val_model.train()

                # ── Checkpoints ───────────────────────────────────────────────
                if step > 0 and step % config.save_interval == 0 and is_main_process():
                    latest_ckpt = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
                    if checkpointer is not None:
                        # Async: D2H snapshot on a copy stream (overlaps the next
                        # step), background thread owns the pickle + disk write.
                        if idle_tracker is not None:
                            idle_tracker.tick("checkpoint")
                        compute_done = torch.cuda.Event() if config.device.startswith("cuda") \
                            else None
                        if compute_done is not None:
                            compute_done.record()
                        ckpt = build_checkpoint_raw(
                            raw_model, optimizer, config, step, best_val_loss, ema=ema
                        )
                        checkpointer.submit_async(
                            latest_ckpt, ckpt, compute_done=compute_done
                        )
                        if idle_tracker is not None:
                            idle_tracker.tick("other")
                    else:
                        save_checkpoint(
                            raw_model, optimizer, config, step, best_val_loss,
                            latest_ckpt, ema=ema,
                        )

    except KeyboardInterrupt:
        # Ctrl-C must not silently discard the run: write latest_checkpoint.pt
        # (and flush any async writes) before propagating the interrupt.
        if is_main_process():
            logger.warning("Interrupted — saving checkpoint before exit")
        ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        if checkpointer is not None:
            compute_done = None
            if config.device.startswith("cuda"):
                compute_done = torch.cuda.Event()
                compute_done.record()
            ckpt = build_checkpoint_raw(
                raw_model, optimizer, config, step, best_val_loss, ema=ema
            )
            checkpointer.submit_async(ckpt_path, ckpt, compute_done=compute_done)
            checkpointer.flush()
        else:
            save_checkpoint(
                raw_model, optimizer, config, step, best_val_loss,
                ckpt_path, ema=ema,
            )
        raise
    # ── Final Save ────────────────────────────────────────────────────────
    total_time = time.time() - t0
    avg_loss = sum(train_losses[-100:]) / min(len(train_losses), 100)

    # Ensure any async checkpoint landed before we exit / write the finale.
    if checkpointer is not None:
        checkpointer.flush()

    if is_main_process():
        final_state = raw_model.state_dict()
        torch.save(final_state, final_model_path)

        if not has_best_model:
            torch.save(final_state, best_model_path)
            logger.info("  best_model.pt seeded from final run (no validation improvement)")
        else:
            logger.info(f"  best_model.pt retained from best val_loss={best_val_loss:.4f}")

        if checkpointer is not None:
            ckpt = build_checkpoint(
                raw_model, optimizer, config, config.max_iters - 1,
                best_val_loss, ema=ema,
            )
            checkpointer.submit(
                os.path.join(checkpoint_dir, "latest_checkpoint.pt"), ckpt
            )
            checkpointer.flush()
        else:
            save_checkpoint(
                raw_model, optimizer, config, config.max_iters - 1, best_val_loss,
                os.path.join(checkpoint_dir, "latest_checkpoint.pt"),
                ema=ema,
            )

        logger.info("=" * 60)
        logger.info("  Training Complete!")
        logger.info(f"  Total time:       {total_time:.1f}s ({total_time/60:.1f} min)")
        logger.info(f"  Final loss:       {avg_loss:.4f}")
        logger.info(f"  Final perplexity: {math.exp(avg_loss):.2f}")
        logger.info(f"  Best val loss:    {best_val_loss:.4f}")
        logger.info(f"  Best val ppl:     {math.exp(best_val_loss):.2f}")
        logger.info(f"  Parameters:       {config.n_params}")
        logger.info(f"  Tokenizer:        {config.tokenizer}")
        logger.info(f"  Checkpoint dir:   {checkpoint_dir}/")
        logger.info("=" * 60)

        _log_wandb({
            "final/loss": avg_loss,
            "final/perplexity": math.exp(avg_loss),
            "final/best_val_loss": best_val_loss,
            "final/total_time_min": total_time / 60,
        }, config.max_iters)

    # Report the overlapped pipeline's GPU idle measurement.
    if idle_tracker is not None and is_main_process():
        st = idle_tracker.stats()
        logger.info(
            f"Pipeline: wall={st['wall_ms']:.0f}ms gpu={st['gpu_ms']:.0f}ms "
            f"idle={st['idle_pct']:.1f}% over {st['steps']} steps"
        )

    # Shut down the pipeline threads (flush any in-flight async checkpoint).
    if checkpointer is not None:
        checkpointer.close()
    if prefetcher is not None:
        prefetcher.stop()

    # Cleanup
    cleanup_ddp()
