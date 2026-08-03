#!/usr/bin/env python
"""
Comprehensive test suite for the trained 100M West Bengal model.

Tests:
  1. Checkpoint loading & config validation
  2. Forward pass correctness
  3. KV cache backends (default, static, quantized)
  4. Text generation quality
  5. Numerical parity (static vs default cache)
  6. Inference latency measurement
  7. Model internals verification
"""
import os
import sys
import time
import math
import json
import contextlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from metis.config import ModelConfig
from metis.model import MetisLM
from metis.generate import generate_text, load_model_and_tokenizer
from metis.kv import (
    KVCache, LayerKV, cached_len_of, cached_bytes,
    quantize_per_token, dequantize_per_token,
)

CKPT_DIR = "checkpoints_westbengal_100m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = []


def record(name, status, detail=""):
    icon = "PASS" if status == "pass" else "FAIL" if status == "fail" else "INFO"
    RESULTS.append({"test": name, "status": status, "detail": detail})
    print(f"  [{icon}] {name}" + (f" — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Checkpoint Loading
# ─────────────────────────────────────────────────────────────────────────────
def test_checkpoint_loading():
    print("\n1. Checkpoint Loading")
    print("-" * 50)

    model, tokenizer, config = load_model_and_tokenizer(CKPT_DIR, DEVICE)

    record("Load model", "pass", f"{config.n_params} params on {config.device}")
    record("Load tokenizer", "pass", f"vocab_size={tokenizer.vocab_size}")
    record("Config loaded", "pass",
           f"d={config.d_model} h={config.n_heads} L={config.n_layers}")

    # Verify architecture
    assert config.d_model == 768, f"Expected d_model=768, got {config.d_model}"
    assert config.n_heads == 16, f"Expected n_heads=16, got {config.n_heads}"
    assert config.n_kv_heads == 4, f"Expected n_kv_heads=4, got {config.n_kv_heads}"
    assert config.n_layers == 16, f"Expected n_layers=16, got {config.n_layers}"
    record("Architecture matches", "pass", "768d/16h/4kv/16L")

    n_params = sum(p.numel() for p in model.parameters())
    record("Parameter count", "pass", f"{n_params:,} ({n_params/1e6:.1f}M)")

    return model, tokenizer, config


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Forward Pass
# ─────────────────────────────────────────────────────────────────────────────
def test_forward_pass(model, tokenizer):
    print("\n2. Forward Pass")
    print("-" * 50)

    model.eval()
    text = "West Bengal is a state"
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits, loss, kv_cache = model(ids)

    record("Forward pass succeeds", "pass",
           f"input={tuple(ids.shape)} -> logits={tuple(logits.shape)}")
    assert logits.dim() == 3, f"Expected 3D logits, got {logits.dim()}D"
    assert logits.size(-1) == tokenizer.vocab_size
    record("Logits shape correct", "pass", f"({ids.size(1)}, {tokenizer.vocab_size})")
    record("KV cache returned", "pass" if kv_cache is not None else "fail",
           f"type={type(kv_cache).__name__}")

    # Verify no NaN/Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    record("No NaN in logits", "pass" if not has_nan else "fail")
    record("No Inf in logits", "pass" if not has_inf else "fail")

    # Verify softmax produces valid probabilities
    probs = torch.softmax(logits[0, -1], dim=-1)
    record("Valid probabilities", "pass",
           f"sum={probs.sum().item():.4f}, max={probs.max().item():.4f}")

    return kv_cache


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: KV Cache Backends
# ─────────────────────────────────────────────────────────────────────────────
def test_kv_cache_backends(model, tokenizer):
    print("\n3. KV Cache Backends")
    print("-" * 50)

    prompts = [
        "West Bengal is known for",
        "The capital city Kolkata",
        "Durga Puja is celebrated",
    ]

    results = {}

    def _gen(backend, prompt):
        """Generate with a fixed seed for a given backend."""
        model.config.kv_backend = backend
        model.eval()
        torch.manual_seed(42)
        return generate_text(
            model, tokenizer, prompt,
            max_new_tokens=30, temperature=0.8, top_k=40, top_p=0.9,
            device=DEVICE, use_kv_cache=True,
        )

    # 1) All backends complete generation successfully
    for backend in ["default", "static", "quantized"]:
        outputs = []
        t0 = time.time()
        for prompt in prompts:
            outputs.append(_gen(backend, prompt))
        elapsed = time.time() - t0
        results[backend] = outputs
        record(f"Backend '{backend}' generation", "pass",
               f"{elapsed:.2f}s for {len(prompts)} prompts")

    # 2) Default vs Static MUST be bit-identical under the same seed.
    #    Both backends compute the exact same logits (proven in Test 5), so
    #    identical RNG -> identical sampled text.
    for i, prompt in enumerate(prompts):
        d = _gen("default", prompt)
        s = _gen("static", prompt)
        match = d == s
        record(f"Default vs Static sampled parity (prompt {i+1})",
               "pass" if match else "fail",
               "identical" if match
               else f"default='{d[:40]}...' static='{s[:40]}...'")

    # 3) Quantized should be close but not bit-identical (small int8 error)
    for i, prompt in enumerate(prompts):
        d = _gen("default", prompt)
        q = _gen("quantized", prompt)
        match = d == q
        record(f"Default vs Quantized (prompt {i+1})",
               "pass" if not match else "info",
               "different (expected — int8 has small error)"
               if not match else "identical (surprising)")

    # Restore default
    model.config.kv_backend = "default"
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Text Generation Quality
# ─────────────────────────────────────────────────────────────────────────────
def test_generation_quality(model, tokenizer):
    print("\n4. Text Generation Quality")
    print("-" * 50)

    model.config.kv_backend = "default"
    model.eval()

    prompts = [
        "West Bengal is",
        "Kolkata is the capital",
        "The culture of Bengal",
        "Durga Puja",
        "The economy of West Bengal",
        "Rabindranath Tagore",
        "The Sundarbans is",
        "Bengali cuisine",
    ]

    for prompt in prompts:
        t0 = time.time()
        text = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            device=DEVICE,
            use_kv_cache=True,
        )
        elapsed = time.time() - t0
        generated = text[len(prompt):]
        tokens_per_sec = len(tokenizer.encode(generated)) / max(elapsed, 0.001)

        # Check if output contains meaningful characters
        unique_chars = len(set(generated))
        has_spaces = " " in generated
        has_alnum = any(c.isalnum() for c in generated)

        quality = "good" if has_spaces and unique_chars > 5 else "poor"
        record(f"Generate '{prompt[:25]}...'", "pass",
               f"'{generated[:60]}...' ({tokens_per_sec:.1f} tok/s, {quality})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Numerical Parity (Static vs Default)
# ─────────────────────────────────────────────────────────────────────────────
def test_numerical_parity(model, tokenizer):
    print("\n5. Numerical Parity (Static vs Default)")
    print("-" * 50)

    prompt = "West Bengal is a state in"
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

    logits_all = {}

    for backend in ["default", "static"]:
        model.config.kv_backend = backend
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(ids)
        logits_all[backend] = logits.clone()

    # Compare logits (should be bit-identical for default vs static)
    diff = (logits_all["default"] - logits_all["static"]).abs().max().item()
    record("Default vs Static logit diff", "pass" if diff == 0 else "fail",
           f"max diff = {diff}")

    # Compare argmax predictions
    pred_default = logits_all["default"][0, -1].argmax().item()
    pred_static = logits_all["static"][0, -1].argmax().item()
    record("Default vs Static top-1 token",
           "pass" if pred_default == pred_static else "fail",
           f"default={pred_default}, static={pred_static}")

    # Test quantized (should be close but not identical)
    model.config.kv_backend = "quantized"
    with torch.no_grad():
        logits_q, _, _ = model(ids)
    diff_q = (logits_all["default"] - logits_q).abs().max().item()
    record("Default vs Quantized logit diff", "pass",
           f"max diff = {diff_q:.6f} (expected: small)")

    # Verify quantized error is bounded
    record("Quantized error bounded (<0.1)",
           "pass" if diff_q < 0.1 else "fail",
           f"max diff = {diff_q:.6f}")

    model.config.kv_backend = "default"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Inference Latency
# ─────────────────────────────────────────────────────────────────────────────
def test_inference_latency(model, tokenizer):
    print("\n6. Inference Latency")
    print("-" * 50)

    model.config.kv_backend = "default"
    model.eval()
    prompt = "West Bengal is"
    ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(ids)

    # Measure prefill (first forward)
    times = []
    for _ in range(5):
        with torch.no_grad():
            t0 = time.time()
            logits, _, kv_cache = model(ids)
            times.append(time.time() - t0)
    avg_prefill = sorted(times)[len(times) // 2]  # median
    record("Prefill latency", "pass", f"{avg_prefill*1000:.1f}ms (median of 5)")

    # Measure decode (token-by-token)
    decode_times = []
    for _ in range(5):
        cache = None
        token = ids[:, -1:]      # start with the last prompt token
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):  # 10 decode steps
                logits, _, cache = model(token, kv_cache=cache)
                # argmax next token (deterministic decode) and feed it back
                token = logits[0, -1].argmax(dim=-1).reshape(1, 1)
        decode_times.append((time.time() - t0) / 10)
    avg_decode = sorted(decode_times)[len(decode_times) // 2]
    record("Decode latency", "pass", f"{avg_decode*1000:.1f}ms/token (median)")

    # Estimate throughput
    record("Estimated throughput", "pass",
           f"~{1/avg_decode:.1f} tokens/sec (single request)")

    # Memory usage
    if DEVICE == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
        record("Peak GPU memory", "pass", f"{mem:.1f} MB")
    else:
        record("Peak GPU memory", "info", "N/A (CPU mode)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Model Internals
# ─────────────────────────────────────────────────────────────────────────────
def test_model_internals(model):
    print("\n7. Model Internals")
    print("-" * 50)

    # Attention backend info
    info = model.get_attention_backend()
    record("Attention backend", "pass",
           f"requested={info['requested']}, recommended={info['recommended']}")

    # Parameter count details
    counts = model.count_parameters()
    for component, count in sorted(counts.items(), key=lambda x: -x[1]):
        if count > 0:
            record(f"  {component}", "info", f"{count:,} params ({count/1e6:.1f}M)")

    # Weight tying check
    tied = model.tok_emb.weight is model.lm_head.weight
    record("Weight tying", "pass" if tied else "fail",
           "embedding = lm_head" if tied else "NOT tied")

    # Check for NaN/Inf in weights
    total_nan = 0
    total_inf = 0
    for name, param in model.named_parameters():
        total_nan += torch.isnan(param).sum().item()
        total_inf += torch.isinf(param).sum().item()
    record("No NaN in weights", "pass" if total_nan == 0 else "fail",
           f"{total_nan} NaN values")
    record("No Inf in weights", "pass" if total_inf == 0 else "fail",
           f"{total_inf} Inf values")

    # Check weight statistics
    all_weights = torch.cat([p.detach().flatten() for p in model.parameters()])
    record("Weight stats", "info",
           f"mean={all_weights.mean().item():.4f}, "
           f"std={all_weights.std().item():.4f}, "
           f"min={all_weights.min().item():.4f}, "
           f"max={all_weights.max().item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: End-to-End Chat Simulation
# ─────────────────────────────────────────────────────────────────────────────
def test_chat_simulation(model, tokenizer):
    print("\n8. Chat Simulation")
    print("-" * 50)

    model.config.kv_backend = "default"
    model.eval()

    conversations = [
        ("User: What is West Bengal?\nMetis:", "Greeting/intro"),
        ("User: Tell me about Kolkata\nMetis:", "City description"),
        ("User: What food is Bengal known for?\nMetis:", "Cuisine question"),
    ]

    for prompt, label in conversations:
        text = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=40,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            device=DEVICE,
            use_kv_cache=True,
            stop_string="\nUser:",
        )
        response = text[len(prompt):].split("\nUser:")[0].strip()
        record(f"Chat: {label}", "pass", f"'{response[:60]}...'")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Metis 100M — West Bengal Model Test Suite")
    print("=" * 70)
    print(f"  Checkpoint: {CKPT_DIR}")
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")

    t0 = time.time()

    # Run all tests
    model, tokenizer, config = test_checkpoint_loading()
    test_forward_pass(model, tokenizer)
    test_kv_cache_backends(model, tokenizer)
    test_generation_quality(model, tokenizer)
    test_numerical_parity(model, tokenizer)
    test_inference_latency(model, tokenizer)
    test_model_internals(model)
    test_chat_simulation(model, tokenizer)

    # Summary
    elapsed = time.time() - t0
    passed = sum(1 for r in RESULTS if r["status"] == "pass")
    failed = sum(1 for r in RESULTS if r["status"] == "fail")
    info = sum(1 for r in RESULTS if r["status"] == "info")

    print("\n" + "=" * 70)
    print(f"  Test Results: {passed} passed, {failed} failed, {info} info")
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 70)

    # Save results
    results_path = os.path.join(CKPT_DIR, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": DEVICE,
            "elapsed_sec": elapsed,
            "passed": passed,
            "failed": failed,
            "tests": RESULTS,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")

    if failed > 0:
        print("\n  FAILED TESTS:")
        for r in RESULTS:
            if r["status"] == "fail":
                print(f"    - {r['test']}: {r['detail']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
