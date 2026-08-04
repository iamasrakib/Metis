"""
Μῆτις (Metis) — Infinite distillation training
===============================================
Trains a Metis model *forever* on text written by a frontier teacher model
reached through an OpenAI-compatible API (see ``metis/teacher.py``).

Loop:
    ask teacher -> it writes a chunk of prose -> Metis trains a few optimizer
    steps on it -> save checkpoint -> repeat, until stopped.

Stopping (every path saves checkpoint + state first, then exits cleanly):
    - a file named ``STOP`` created in the checkpoint dir
    - Ctrl+C (KeyboardInterrupt)
    - ``--max-steps N`` reached
    - ``--budget-tokens T`` total teacher tokens reached (0 = unlimited)

Resume: re-running the same command with the same ``--checkpoint-dir``
reloads ``latest_checkpoint.pt`` + ``distill_state.json`` + ``tokenizer.json``
and continues exactly where it left off — no manual setup.

Design notes
------------
- Text distillation: the teacher writes prose and Metis trains on it with its
  normal next-token loss. True logit-level distillation is impossible across
  different tokenizers (Metis char/BPE vs the teacher's own), so this is the
  standard, buildable meaning of "distill from an API".
- Simple eager training step (no overlapped pipeline / CUDA graphs / packing).
  The teacher API call is the bottleneck (seconds per chunk), so step
  throughput is irrelevant. Keeps the loop easy to audit.
- Mixed precision reuses ``get_amp_dtype``: T4/older GPUs train in fp16 (bf16
  has no fused SDPA kernel on Turing and crashes), Ampere+ in bf16, CPU fp32.
- The tokenizer is fit ONCE on first run (or built from a fixed BPE encoding)
  and saved to ``tokenizer.json``. On resume it is loaded, never re-fit — an
  endless stream must not drift the vocabulary.
- Teacher text accumulates in a rolling token buffer (``_TokenStream``) that
  is drained into fixed-size micro-batches, so batch shape never depends on
  how many tokens one API call happens to return.
"""

import json
import logging
import os
import time
from dataclasses import dataclass

import torch

from .config import ModelConfig, get_amp_dtype
from .data import BPETokenizer, CharTokenizer
from .model import MetisLM
from .teacher import MockTeacher, TeacherError, build_teacher
from .training import get_lr, load_checkpoint, save_checkpoint

logger = logging.getLogger("metis.distill")

STOP_FILENAME = "STOP"
STATE_FILENAME = "distill_state.json"
CONFIG_FILENAME = "config.json"
TOKENIZER_FILENAME = "tokenizer.json"
TEXT_LOG_FILENAME = "distill_text.log"
CHECKPOINT_FILENAME = "latest_checkpoint.pt"

_TEXT_LOG_CAP = 2_000_000          # bytes kept of the rolling text log
_STREAM_TOKEN_CAP = 1_000_000      # max live tokens in the rolling buffer

DEFAULT_SYSTEM_PROMPT = (
    "You are a teacher writing plain, factual prose for a small language "
    "model to train on. Write in simple English sentences. Do not use "
    "markdown, bullet lists, headings, or dialogue. Just continuous "
    "paragraphs of informative text."
)


# ──────────────────────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DistillOptions:
    """Distillation-only knobs. Kept OUT of ModelConfig on purpose: the
    serialized config schema stays untouched, so distill-trained checkpoint
    dirs remain fully compatible with ``metis chat`` / ``generate``."""

    topic: str = "general knowledge"
    topic_file: str | None = None          # one topic per line, rotated through
    seed_data: str | None = None           # corpus to fit a char tokenizer once
    max_steps: int = 0                     # 0 = run forever
    save_every: int = 50                   # steps between checkpoint+state saves
    steps_per_call: int = 4                # optimizer steps per teacher API call
    max_tokens: int = 1024                 # max tokens per teacher call
    min_sleep: float = 1.0                 # min seconds between calls (cost guard)
    budget_tokens: int = 0                 # stop after N total teacher tokens (0 = no)
    no_resume: bool = False                # ignore existing checkpoint/state
    mock: bool = False                     # offline mock teacher (no network)
    test_teacher: bool = False             # one connectivity call, then exit
    teacher_base_url: str | None = None    # flag > env > default (see teacher.py)
    teacher_api_key: str | None = None
    teacher_model: str | None = None
    teacher_timeout: int = 240             # seconds per teacher call


# ──────────────────────────────────────────────────────────────────────────────
# Persistent run state
# ──────────────────────────────────────────────────────────────────────────────

class DistillState:
    """Metadata that must survive a restart, beyond the model checkpoint."""

    def __init__(self, path: str):
        self.path = path
        self.step = 0                  # last completed optimizer step
        self.tokens_seen = 0           # teacher tokens added to the stream
        self.teacher_tokens = 0        # API tokens used (usage.total_tokens)
        self.api_calls = 0             # teacher API calls made
        self.topic_index = 0           # rotation cursor into the topic list
        self.started_ts = 0.0
        self.updated_ts = 0.0

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            logger.info(
                f"Distill state loaded <- {self.path} "
                f"(step {self.step}, teacher_tokens {self.teacher_tokens})"
            )

    def save(self) -> None:
        payload = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "teacher_tokens": self.teacher_tokens,
            "api_calls": self.api_calls,
            "topic_index": self.topic_index,
            "started_ts": self.started_ts,
            "updated_ts": self.updated_ts,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_topics(opts: DistillOptions) -> list[str]:
    if opts.topic_file:
        with open(opts.topic_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise TeacherError(f"Topic file {opts.topic_file!r} is empty.")
        return lines
    return [opts.topic]


def _next_topic(topics: list[str], state: DistillState) -> str:
    topic = topics[state.topic_index % len(topics)]
    state.topic_index += 1
    return topic


def _topic_user_prompt(topic: str) -> str:
    return (
        f"Write informative, plain prose about: {topic}. "
        f"Produce several connected paragraphs."
    )


def _stop_reason(ckpt_dir: str, opts: DistillOptions, step: int,
                 state: DistillState) -> str | None:
    # ``step`` here is the NEXT step to run (1-based). Stop once we have
    # completed ``max_steps``, i.e. when the next step exceeds it.
    if opts.max_steps and step > opts.max_steps:
        return f"reached --max-steps {opts.max_steps}"
    if opts.budget_tokens and state.teacher_tokens >= opts.budget_tokens:
        return f"reached --budget-tokens {opts.budget_tokens}"
    stop_path = os.path.join(ckpt_dir, STOP_FILENAME)
    if os.path.exists(stop_path):
        return f"STOP file present: {stop_path}"
    return None


def _reload_saved_config(config: ModelConfig, opts: DistillOptions) -> ModelConfig:
    """On resume, adopt the checkpoint's architecture but keep runtime overrides.

    The user may start with different flags than the original run; the model
    dims must still match the saved weights. Runtime fields (device, attention
    / KV backend, log level) keep the values this invocation asked for.
    """
    config_path = os.path.join(config.checkpoint_dir, CONFIG_FILENAME)
    if opts.no_resume or not os.path.exists(config_path):
        return config
    saved = ModelConfig.from_json(config_path)
    for attr in ("device", "attn_backend", "kv_backend", "log_level",
                 "checkpoint_dir"):
        setattr(saved, attr, getattr(config, attr))
    logger.info(f"Resuming with architecture from {config_path}")
    return saved


def _make_tokenizer(name: str) -> CharTokenizer | BPETokenizer:
    if name == "char":
        return CharTokenizer()
    return BPETokenizer(encoding_name=name)


def _build_or_load_tokenizer(config: ModelConfig, opts: DistillOptions):
    """Return a tokenizer whose vocabulary is STABLE across restarts.

    BPE encodings (e.g. cl100k_base) have a fixed tiktoken vocab — always
    rebuilt identically, never re-fit. A char tokenizer is fit once (from
    ``--seed-data``, or a minimal fallback alphabet) and saved; on resume it
    is loaded from disk, never re-fit, so the stream cannot drift the vocab.
    """
    tokenizer_path = os.path.join(config.checkpoint_dir, TOKENIZER_FILENAME)
    if not opts.no_resume and os.path.exists(tokenizer_path):
        # Same detection as generate.py's load_model_and_tokenizer: the saved
        # file decides the type, and load() reconstructs the exact vocab.
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            header = json.load(f)
        tok = BPETokenizer() if header.get("type") == "bpe" else CharTokenizer()
        tok.load(tokenizer_path)
        logger.info("Tokenizer loaded from checkpoint dir (not re-fit).")
        config.vocab_size = tok.vocab_size
        return tok

    tok = _make_tokenizer(config.tokenizer)
    seed_text = ""
    if opts.seed_data:
        with open(opts.seed_data, "r", encoding="utf-8") as f:
            seed_text = f.read()
    if seed_text:
        tok.fit(seed_text)
    elif config.tokenizer == "char":
        # No seed corpus: fit a minimal alphabet so the run works, but warn.
        # Out-of-vocab chars map to <unk> (safe), so nothing crashes.
        logger.warning(
            "char tokenizer without --seed-data: the vocab is a minimal "
            "fallback alphabet, so most real text becomes <unk>. Prefer "
            "--tokenizer cl100k_base or --seed-data <corpus.txt>."
        )
        tok.fit("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789.,;:!?-'\n")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tok.save(tokenizer_path)
    config.vocab_size = tok.vocab_size
    return tok


class _TokenStream:
    """Rolling token buffer fed by the teacher, drained into micro-batches.

    The buffer accumulates unconsumed tokens across API calls, so a step's
    batch shape never depends on how many tokens one call returns. A capped
    live window bounds memory if training ever falls behind the API.
    """

    def __init__(self, cap: int = _STREAM_TOKEN_CAP):
        self._buf: list[int] = []
        self._offset = 0
        self._cap = cap

    def extend(self, ids: list[int]) -> None:
        self._buf.extend(ids)

    def _compact(self) -> None:
        live = self._buf[self._offset:]
        if len(live) > self._cap:
            live = live[-self._cap:]
        self._buf = list(live)
        self._offset = 0

    def try_step_batches(self, seq: int, micro_batch: int,
                         grad_accum: int) -> tuple[list, int]:
        """If enough tokens are buffered, build one step's micro-batches.

        Returns ``(micro_batches, tokens_consumed)``; empty list when the
        buffer is too small for a full step.
        """
        avail = len(self._buf) - self._offset
        need = micro_batch * grad_accum * (seq + 1)
        if avail < need:
            return [], 0
        batches = []
        for _ in range(grad_accum):
            x = torch.zeros(micro_batch, seq, dtype=torch.long)
            y = torch.zeros(micro_batch, seq, dtype=torch.long)
            for b in range(micro_batch):
                seg = self._buf[self._offset:self._offset + seq + 1]
                x[b] = torch.tensor(seg[:seq], dtype=torch.long)
                y[b] = torch.tensor(seg[1:seq + 1], dtype=torch.long)
                self._offset += seq
            batches.append((x, y))
        consumed = micro_batch * grad_accum * seq
        self._compact()
        return batches, consumed


def _append_text_log(ckpt_dir: str, text: str, topic: str) -> None:
    """Append raw teacher text to a capped log so training content is inspectable."""
    path = os.path.join(ckpt_dir, TEXT_LOG_FILENAME)
    try:
        if os.path.exists(path) and os.path.getsize(path) > _TEXT_LOG_CAP:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(os.path.getsize(path) - _TEXT_LOG_CAP // 2)
                tail = f.read()
            with open(path, "w", encoding="utf-8") as f:
                f.write("[log truncated - keeping newest text]\n\n" + tail)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n===== {topic} =====\n{text}\n")
    except OSError:
        pass  # text logging is best-effort


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────────

def _optimizer_step(model: MetisLM, optimizer, scaler, config: ModelConfig,
                    micro_batches: list, amp_dtype, device: str, use_amp: bool,
                    step: int) -> tuple[float, float]:
    """One optimizer step over ``grad_accum`` micro-batches (mirrors the eager
    step in metis/training.py, minus pipeline/CUDA-graph plumbing)."""
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    use_grad_ckpt = device.startswith("cuda")
    for (x, y) in micro_batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0],
                            dtype=amp_dtype, enabled=use_amp):
            _, loss, _ = model(x, y, use_checkpointing=use_grad_ckpt)
            loss = loss / len(micro_batches)
        scaler.scale(loss).backward()
        loss_accum += loss.item()

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.max_grad_norm
    )
    scaler.step(optimizer)
    scaler.update()
    model.invalidate_moe_caches()
    return loss_accum, grad_norm


# ──────────────────────────────────────────────────────────────────────────────
# The loop
# ──────────────────────────────────────────────────────────────────────────────

def _save_snapshot(ckpt_dir: str, model: MetisLM, optimizer, config: ModelConfig,
                   step: int, state: DistillState) -> None:
    ckpt_path = os.path.join(ckpt_dir, CHECKPOINT_FILENAME)
    save_checkpoint(model, optimizer, config, step, float("inf"), ckpt_path)
    state.save()
    logger.info(f"Checkpoint + state saved (step {step})")


def distill(config: ModelConfig, opts: DistillOptions) -> int:
    """Run the infinite distillation loop. Returns process exit code."""
    from .config import setup_logging
    setup_logging(config.log_level, config.log_dir)
    ckpt_dir = config.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    # 0. Resume the architecture from the checkpoint if present.
    config = _reload_saved_config(config, opts)

    # 1. Teacher.
    if opts.mock:
        teacher = MockTeacher(topics=tuple(_load_topics(opts)))
        logger.info("Using MOCK teacher (offline, no API).")
    else:
        teacher = build_teacher(opts)
        logger.info(
            f"Teacher: {teacher.model} @ {teacher.base_url}"
        )

    if opts.test_teacher:
        try:
            reply = teacher.complete(
                "You are a connectivity test. Reply with exactly the word OK.",
                "Connectivity check.",
                max_tokens=8,
                temperature=0.0,
            )
        except TeacherError as e:
            logger.error(f"Teacher connectivity test FAILED: {e}")
            return 1
        print(f"Teacher OK: {reply!r}")
        print("If this looks right, start distillation:")
        print(f"  metis distill --checkpoint-dir {ckpt_dir}")
        return 0

    # 2. Tokenizer (fit once, load forever).
    tokenizer = _build_or_load_tokenizer(config, opts)

    # 3. Model + optimizer + resume weights.
    logger.info(
        f"Building model: {config.n_params if config.n_params != 'unknown' else '?'} "
        f"params, vocab {config.vocab_size}, device {config.device}"
    )
    model = MetisLM(config)
    model.to(config.device)
    optimizer = model.configure_optimizers(
        config.weight_decay, config.learning_rate, config.device
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.device.startswith("cuda"))
    config.save_json(os.path.join(ckpt_dir, CONFIG_FILENAME))

    state = DistillState(os.path.join(ckpt_dir, STATE_FILENAME))
    state.started_ts = time.time()
    ckpt_step = -1
    ckpt_path = os.path.join(ckpt_dir, CHECKPOINT_FILENAME)
    if not opts.no_resume and os.path.exists(ckpt_path):
        loaded = load_checkpoint(ckpt_path, model, optimizer, config.device)
        ckpt_step = loaded["step"]
        logger.info("Weights + optimizer resumed from latest_checkpoint.pt")
    state.load()
    start_step = max(ckpt_step, state.step) + 1
    logger.info(
        f"Distillation starting at step {start_step} "
        f"(runs forever until you stop it)."
    )

    # 4. Infinite loop.
    topics = _load_topics(opts)
    stream = _TokenStream()
    model.train()
    amp_dtype = get_amp_dtype(config.device)
    use_amp = config.device.startswith("cuda")
    device = config.device
    step = start_step - 1
    t0 = time.time()
    exit_code = 0

    try:
        while True:
            reason = _stop_reason(ckpt_dir, opts, step + 1, state)
            if reason:
                logger.info(f"Stopping: {reason}")
                break

            topic = _next_topic(topics, state)
            logger.info(
                f"[call {state.api_calls}] teacher writing about: {topic}"
            )
            text = teacher.complete(
                DEFAULT_SYSTEM_PROMPT,
                _topic_user_prompt(topic),
                opts.max_tokens,
            )
            state.api_calls += 1
            usage = getattr(teacher, "last_total_tokens", None)
            state.teacher_tokens += usage if usage else _approx_tokens(text)
            state.updated_ts = time.time()
            _append_text_log(ckpt_dir, text, topic)

            ids = tokenizer.encode(text)
            stream.extend(ids)
            state.tokens_seen += len(ids)

            # Drain as many full optimizer steps as the buffer allows, up to
            # steps_per_call. If the buffer is too small, wait for the next
            # API call (this naturally paces training to teacher throughput).
            steps_this_call = 0
            while steps_this_call < opts.steps_per_call:
                reason = _stop_reason(ckpt_dir, opts, step + 1, state)
                if reason:
                    break
                batches, _ = stream.try_step_batches(
                    config.max_seq_len,
                    config.micro_batch_size,
                    config.gradient_accumulation_steps,
                )
                if not batches:
                    break
                loss_accum, grad_norm = _optimizer_step(
                    model, optimizer, scaler, config, batches,
                    amp_dtype, device, use_amp, step + 1,
                )
                step += 1
                steps_this_call += 1
                state.step = step

                if step % opts.save_every == 0:
                    _save_snapshot(ckpt_dir, model, optimizer, config, step, state)
                if step % config.log_interval == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {step} | loss {loss_accum:.4f} | "
                        f"lr {_current_lr(optimizer):.2e} | "
                        f"grad {grad_norm:.2f} | teacher_tokens "
                        f"{state.teacher_tokens} | {elapsed:.0f}s"
                    )

            if opts.min_sleep > 0:
                time.sleep(opts.min_sleep)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt - saving state and stopping.")
    finally:
        # Only snapshot when at least one step actually ran (step starts at 0
        # on a fresh dir; a STOP-before-any-work run must not write a bogus
        # "step 0" checkpoint that resume would treat as real progress).
        if step > 0:
            _save_snapshot(ckpt_dir, model, optimizer, config, step, state)
        logger.info(
            f"Stopped at step {step}. Re-run the same command anytime to "
            f"resume - no setup needed."
        )

    return exit_code


def _current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]
