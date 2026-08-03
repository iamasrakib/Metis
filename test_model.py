#!/usr/bin/env python3
"""
Metis — Simple Model Tester CLI
================================

One easy command to test the trained Metis model. Everything is ASCII-only so
it works on any Windows console / codepage.

    python test_model.py info
    python test_model.py generate --prompt "Why do cats purr?"
    python test_model.py chat
    python test_model.py test

On Windows you can just use the launcher (it picks the right Python for you):

    test_model.bat generate --prompt "Why do cats purr?"

Note: this project's Python lives in the global Python 3.11 install (torch +
metis are installed there). The repo's .venv is currently EMPTY, so plain
`python test_model.py` may fail with "No module named 'torch'". Use
`test_model.bat`, or run this script with:
    C:\\Users\\iamas\\AppData\\Local\\Programs\\Python\\Python311\\python.exe test_model.py ...
"""

import argparse
import json
import os
import subprocess
import sys


def _die(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def _require_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ── info ──────────────────────────────────────────────────────────────────────

def cmd_info(args: argparse.Namespace) -> int:
    ckpt = args.checkpoint_dir
    if not os.path.isdir(ckpt):
        return _die(f"Checkpoint dir not found: {os.path.abspath(ckpt)}\n"
                    "  Train a model first: metis train --dataset data/input.txt")

    cfg_path = os.path.join(ckpt, "config.json")
    print("Metis model info")
    print("-" * 48)
    if os.path.exists(cfg_path):
        from metis import ModelConfig
        cfg = ModelConfig.from_json(cfg_path)
        # n_params is a non-init field (set at model build time, not from_json),
        # so read the recorded value straight from the saved JSON.
        try:
            with open(cfg_path, encoding="utf-8") as f:
                n_params = json.load(f).get("n_params", "unknown")
        except Exception:
            n_params = "unknown"
        print(f"  Architecture : {cfg.d_model} dim / {cfg.n_layers} layers / "
              f"{cfg.n_heads} heads / {cfg.n_kv_heads} KV heads")
        print(f"  Parameters   : {n_params}")
        print(f"  Tokenizer    : {cfg.tokenizer} (vocab {cfg.vocab_size})")
        print(f"  Context      : {cfg.max_seq_len} tokens")
        flags = []
        if cfg.use_moe:
            flags.append(f"MoE {cfg.moe_num_experts} experts, top-{cfg.moe_top_k}")
        if cfg.use_qk_norm:
            flags.append("QK-norm")
        if cfg.use_attention_sink:
            flags.append("attention sink")
        if cfg.use_rope:
            flags.append("RoPE")
        if flags:
            print(f"  Extras       : {', '.join(flags)}")
        try:
            from metis.attn import detect_attention_backends
            be = detect_attention_backends()
            req = getattr(cfg, "attn_backend", "auto")
            print(f"  Attention    : requested={req}, active={be['recommended']} "
                  f"(torch-flash={be['torch_flash']}, "
                  f"mem-eff={be['torch_mem_efficient']})")
        except Exception:
            pass
    else:
        print("  config.json  : MISSING")
    for name in ("best_model.pt", "final_model.pt", "latest_checkpoint.pt"):
        p = os.path.join(ckpt, name)
        if os.path.exists(p):
            print(f"  {name:<18}: {os.path.getsize(p) / 1e6:.1f} MB")
    return 0


# ── generate ──────────────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> int:
    if not _require_torch():
        return _die("ERROR: PyTorch not installed in this Python.\n"
                    "  Use test_model.bat, or run with the Python that has torch:\n"
                    "    C:\\Users\\iamas\\AppData\\Local\\Programs\\Python\\Python311\\python.exe test_model.py ...")

    from metis.generate import generate_text, load_model_and_tokenizer

    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint_dir, args.device)
    print(f"Device: {config.device} | Prompt: {args.prompt!r}")
    print("-" * 48)

    out = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=config.device,
    )
    if out.startswith(args.prompt):
        out = out[len(args.prompt):]
    print(out.strip())
    return 0


# ── chat ──────────────────────────────────────────────────────────────────────

def _newline_token_id(tokenizer) -> int | None:
    """Find the token id for '\\n', tolerant of both tokenizer types."""
    stoi = getattr(tokenizer, "stoi", None)
    if isinstance(stoi, dict) and "\n" in stoi:
        return stoi["\n"]
    try:
        ids = tokenizer.encode("\n")
        if ids:
            return ids[0]
    except Exception:
        pass
    return None


def cmd_chat(args: argparse.Namespace) -> int:
    if not _require_torch():
        return _die("ERROR: PyTorch not installed in this Python.\n"
                    "  Use test_model.bat, or run with the Python that has torch:\n"
                    "    C:\\Users\\iamas\\AppData\\Local\\Programs\\Python\\Python311\\python.exe test_model.py ...")

    from metis.generate import generate_text, load_model_and_tokenizer

    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint_dir, args.device)
    stop_token = _newline_token_id(tokenizer)
    temperature = args.temp

    print("Metis chat  (commands: /quit, /temp 0.5)")
    print("-" * 48)
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input:
            continue
        low = user_input.lower()
        if low in ("/quit", "/exit", "/q"):
            print("Bye!")
            break
        if user_input.startswith("/temp"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"[temperature set to {temperature}]")
                    continue
                except ValueError:
                    pass
            print("[usage: /temp 0.5]")
            continue
        if user_input.startswith("/"):
            print("[commands: /quit | /temp 0.5]")
            continue

        prompt = f"User: {user_input}\nMetis:"
        out = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=args.max_tokens,
            temperature=temperature,
            top_k=40, top_p=0.9, repetition_penalty=1.1,
            device=config.device,
            stop_token_id=stop_token,
        )
        reply = out[len(prompt):].strip().split("\n")[0]
        print(f"Metis> {reply}\n")


# ── test ──────────────────────────────────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    print("Running built-in test suite: " + " ".join(cmd))
    print("-" * 48)
    return subprocess.call(cmd, cwd=root)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="test_model",
        description="Simple CLI to test the trained Metis model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  test_model.py generate --prompt "Why do cats purr?"\n'
            "  test_model.py chat\n"
            "  test_model.py info\n"
            "  test_model.py test\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="Show model status")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("generate", help="Generate text from a prompt")
    p.add_argument("--prompt", required=True, help="Text to continue from")
    p.add_argument("--max-tokens", type=int, default=150, help="Max tokens to generate (default: 150)")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature; 0 = greedy (default: 0.7)")
    p.add_argument("--top-k", type=int, default=40, help="Top-k sampling (default: 40)")
    p.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling (default: 0.9)")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty (default: 1.1)")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory (default: checkpoints)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("chat", help="Interactive chat")
    p.add_argument("--temp", type=float, default=0.7, help="Starting temperature (default: 0.7)")
    p.add_argument("--max-tokens", type=int, default=150, help="Max tokens per reply (default: 150)")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory (default: checkpoints)")
    p.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("test", help="Run the built-in test suite")
    p.set_defaults(func=cmd_test)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
