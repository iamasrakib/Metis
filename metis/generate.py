"""
Μῆτις (Metis) — Text Generation
==================================
Professional inference pipeline with:
  • KV-cache for efficient autoregressive generation
  • Temperature, top-k, top-p (nucleus) sampling
  • Repetition penalty
  • Streaming output (token-by-token)
  • Interactive chat mode with conversation history
  • CLI interface with full control over generation parameters
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable

import torch
from torch.nn import functional as F

from .config import ModelConfig, get_amp_dtype
from .data import BPETokenizer, CharTokenizer
from .kv import cached_len_of  # KV cache subsystem (Phase 7)
from .model import MetisLM

logger = logging.getLogger("metis.generate")


# ──────────────────────────────────────────────────────────────────────────────
# Core Generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_text(
    model: MetisLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.0,
    device: str = "cpu",
    stop_token_id: int | None = None,
    stop_string: str | None = None,
    stream_callback: Callable[[str], None] | None = None,
    use_kv_cache: bool = True,
) -> str:
    """Generate text autoregressively from a prompt.

    Args:
        model: Trained MetisLM model.
        tokenizer: Fitted tokenizer.
        prompt: Starting text.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_k: Keep only top-k logits (None = disabled).
        top_p: Nucleus sampling threshold (None = disabled).
        repetition_penalty: Penalty for repeated tokens (1.0 = disabled).
        device: Compute device.
        stop_token_id: Stop when this token is generated.
        stop_string: Stop when this string appears in output.
        stream_callback: Called with each new token string for streaming.
        use_kv_cache: Use KV-cache for faster generation.

    Returns:
        Full generated text (prompt + generated tokens).
    """
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # Optional execution scheduler: wraps the model in an arena-reuse infer path
    # when the env var is set (via config or CLI --exec-scheduler).
    _scheduler = None
    if os.environ.get("METIS_EXEC_SCHEDULER", "0") == "1":
        try:
            from .scheduler import INFER, build_scheduler
            _scheduler = build_scheduler(model, mode=INFER, device=device,
                                         calibrate_run=False)
        except Exception as e:
            logger.warning("exec scheduler unavailable: %s", e)

    _exec = _scheduler.execute if _scheduler is not None else None
    generated_ids = []
    kv_cache = None

    # AMP inference: on CUDA the forward runs under bf16/fp16 autocast so the
    # fused attention kernels (FlashAttention / memory-efficient) engage; the
    # logits are cast back to fp32 before sampling so sampling stays exact.
    use_amp = device.startswith("cuda")
    amp_dtype = get_amp_dtype(device)

    for step in range(max_new_tokens):
        # Prepare input. With a KV-cache we normally feed only the last token,
        # but if the cached length is about to exceed max_seq_len we must reset
        # the cache and re-prefill from the last max_seq_len tokens (sliding
        # window). Otherwise the RoPE position offset would index past the
        # precomputed frequency buffer.
        if use_kv_cache and kv_cache is not None:
            # Cache-aware live length: works for the legacy list of (K, V)
            # tuples, a KVCache (static/quantized), and MLA caches alike.
            cached_len = cached_len_of(kv_cache)
            if cached_len + 1 > model.config.max_seq_len:
                # Sliding window: reuse the buffers (static/quantized) or drop
                # and re-process the tail (legacy / MLA).
                if hasattr(kv_cache, "reset"):
                    kv_cache.reset()
                else:
                    kv_cache = None
                input_ids = idx[:, -model.config.max_seq_len:]
            else:
                input_ids = idx[:, -1:]
        else:
            input_ids = idx[:, -model.config.max_seq_len:]
            kv_cache = None

        # Forward pass (autocast on CUDA → fused attention kernels)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            _forward = _exec if _exec is not None else model
            logits, _, kv_cache = _forward(input_ids, kv_cache=kv_cache if use_kv_cache else None)
        logits = logits.float()[:, -1, :]  # (B, vocab_size), fp32 for sampling

        # Temperature scaling
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy decoding
            idx_next = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
            generated_ids.append(idx_next.item())
            token_str = tokenizer.decode([idx_next.item()], skip_special=False)
            if stream_callback:
                stream_callback(token_str)
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                break
            continue

        # Repetition penalty
        if repetition_penalty != 1.0 and len(generated_ids) > 0:
            for prev_id in set(generated_ids[-50:]):  # Look back 50 tokens
                if logits[0, prev_id] > 0:
                    logits[0, prev_id] /= repetition_penalty
                else:
                    logits[0, prev_id] *= repetition_penalty

        # Top-k filtering
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            top_values, _ = torch.topk(logits, k)
            logits[logits < top_values[:, [-1]]] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, idx_next], dim=1)
        generated_ids.append(idx_next.item())

        # Stream output
        token_str = tokenizer.decode([idx_next.item()], skip_special=False)
        if stream_callback:
            stream_callback(token_str)

        # Stop conditions
        if stop_token_id is not None and idx_next.item() == stop_token_id:
            break
        if stop_string is not None:
            decoded_so_far = tokenizer.decode(generated_ids)
            if stop_string in decoded_so_far:
                break

    return tokenizer.decode(idx[0].tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Interactive Chat
# ──────────────────────────────────────────────────────────────────────────────

def chat(
    model: MetisLM,
    tokenizer: CharTokenizer,
    config: ModelConfig,
    stream: bool = True,
) -> None:
    """Interactive chat loop with streaming output."""

    BANNER = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   Μῆτις (Metis) — Interactive Chat                       ║
║                                                          ║
║   Commands:                                              ║
║     /quit or /exit  — End the conversation               ║
║     /clear          — Clear conversation history         ║
║     /help           — Show this message                  ║
║     /temp <value>   — Set temperature (e.g. /temp 0.5)   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(BANNER)

    temperature = 0.8
    stop_token_id = tokenizer.stoi.get("\n", None)

    while True:
        try:
            user_input = input("\n  You ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/quit", "/exit"):
                print("\nGoodbye! 👋")
                break
            elif cmd[0] == "/clear":
                print("  [Conversation cleared]")
                continue
            elif cmd[0] == "/help":
                print(BANNER)
                continue
            elif cmd[0] == "/temp" and len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    print(f"  [Temperature set to {temperature}]")
                except ValueError:
                    print("  [Invalid temperature value]")
                continue
            else:
                print(f"  [Unknown command: {cmd[0]}]")
                continue

        # Format prompt
        prompt = f"User: {user_input}\nMetis:"

        # Stream response
        print("\n  Metis ❯ ", end="", flush=True)

        def stream_token(token: str):
            if token != "\n":
                print(token, end="", flush=True)

        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=150,
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.1,
            device=config.device,
            stop_token_id=stop_token_id,
            stream_callback=stream_token if stream else None,
        )

        if not stream:
            response = generated[len(prompt):].strip().split("\n")[0]
            print(response)
        else:
            print()  # Newline after streamed output


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    checkpoint_dir: str = "checkpoints",
    device: str | None = None,
    attn_backend: str | None = None,
    kv_backend: str | None = None,
) -> tuple:
    """Load a trained model and tokenizer from checkpoint directory.

    Args:
        checkpoint_dir: Directory with config.json / tokenizer.json / weights.
        device: Device override (cpu/cuda); auto-detected if None.
        attn_backend: Override the saved attention backend
            ("auto"|"flash_attn"|"sdpa"|"flash"|"mem_efficient"|"math").
        kv_backend: Override the saved KV-cache backend
            ("default"|"static"|"quantized"|"mla") — the KV subsystem
            (Phase 7). ``None`` falls back to the saved config / env var.

    Returns:
        Tuple of (model, tokenizer, config).
    """
    config = ModelConfig()
    if device:
        config.device = device

    # Load tokenizer — detect format from the JSON
    tokenizer: BPETokenizer | CharTokenizer
    json_path = os.path.join(checkpoint_dir, "tokenizer.json")
    pkl_path = os.path.join(checkpoint_dir, "tokenizer.pkl")

    if os.path.exists(json_path):
        # Auto-detect tokenizer type from JSON
        with open(json_path) as f:
            tok_data = json.load(f)
        tok_type = tok_data.get("type", "char")
        tok_version = tok_data.get("version", "2.0")

        if tok_type == "bpe" and tok_version == "3.0":
            tokenizer = BPETokenizer()
        else:
            tokenizer = CharTokenizer()
        tokenizer.load(json_path)
    elif os.path.exists(pkl_path):
        tokenizer = CharTokenizer()
        tokenizer.load(pkl_path)
    else:
        print("❌ Tokenizer not found. Please train the model first.")
        print(f"   Expected: {json_path} or {pkl_path}")
        sys.exit(1)

    config.vocab_size = tokenizer.vocab_size

    # Load saved config if available
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        saved_config = ModelConfig.from_json(config_path)
        # Preserve device from CLI
        saved_device = config.device
        config = saved_config
        config.device = saved_device
        config.vocab_size = tokenizer.vocab_size

    # Optional attention-backend override (CLI / explicit request)
    if attn_backend is not None:
        config.attn_backend = attn_backend

    # Optional KV-cache backend override (CLI > env var > saved config)
    if kv_backend is not None:
        config.kv_backend = kv_backend
    elif os.environ.get("METIS_KV_BACKEND"):
        config.kv_backend = os.environ["METIS_KV_BACKEND"]

    # Load model
    model = MetisLM(config)
    model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    if not os.path.exists(model_path):
        print("❌ Model checkpoint not found. Please train the model first.")
        print("   Run: python train.py --dataset data/input.txt")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location=config.device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.to(config.device)
    model.eval()

    logger.info(f"Model loaded from {model_path} ({config.n_params} parameters)")
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(
        description="Μῆτις (Metis) — Generate text or chat with the model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate.py                              # Interactive chat\n"
            '  python generate.py --prompt "Hello world"       # Single generation\n'
            "  python generate.py --temperature 0.5 --top-k 20 # Adjust sampling\n"
        ),
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="Generate from this prompt (non-interactive mode)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k sampling (default: 40)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory (default: checkpoints)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cpu/cuda)")
    parser.add_argument("--kv-backend", type=str, default=None, dest="kv_backend",
                        choices=["default", "static", "quantized", "mla"],
                        help="KV cache backend (default: saved config / "
                             "METIS_KV_BACKEND env): default = legacy growable, "
                             "static = preallocated buffers (bit-identical), "
                             "quantized = int8 compressed, mla = latent attention")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming output")

    args = parser.parse_args()
    model, tokenizer, config = load_model_and_tokenizer(
        args.checkpoint_dir, args.device, kv_backend=args.kv_backend)

    if args.prompt:
        # Single generation mode
        print(f"Prompt: {args.prompt}\n")
        generated = generate_text(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=config.device,
        )
        print(generated)
    else:
        # Interactive chat mode
        chat(model, tokenizer, config, stream=not args.no_stream)


if __name__ == "__main__":
    main()
