"""
Μῆτις (Metis) — Operator cost model for the execution scheduler
================================================================
Fills each graph node's ``est_ms`` with a roofline estimate:

    est_ms = max(flops / peak_ops, bytes / bandwidth) * 1000

The two terms model the compute-bound and memory-bound regimes: a GEMM's FLOPs
dominate, an elementwise op's byte traffic dominates. Peak constants are
device-dependent and **measured at startup** for the current device with a
one-time micro-benchmark (:func:`probe_bandwidth`), so the estimate reflects
this machine (e.g. CPU DRAM bandwidth vs a CUDA HBM). Explicit CUDA constants
are used when probing is unavailable, and everything can be overridden with
``METIS_PEAK_OPS`` / ``METIS_BANDWIDTH`` (giga-units / GB/s).

The result is *relative* accuracy — good enough to order the critical path and
to compare operators — not a kernel-timer. ``calibrate`` can optionally time
each node kind once against the real model and rescale the whole plan to the
measured scale.
"""

from __future__ import annotations

import os

import torch

# Roofline constants (giga-ops/s, GB/s).  Keys are "cpu" and "cuda".
# CPU: measured at startup by :func:`probe_bandwidth`; these are the fallback.
# CUDA: representative HBM/FP16-tensor-core figures; the startup probe
# overrides them on the actual device.
_DEFAULT_PEAK_OPS = {"cpu": 20e9, "cuda": 120e12}
_DEFAULT_BANDWIDTH = {"cpu": 25e9, "cuda": 900e9}

# Node kinds whose cost is dominated by memory traffic rather than arithmetic.
_MEMORY_BOUND = frozenset({
    "embed", "embed_pos", "drop", "norm", "rope", "silu_mul", "act",
    "contig", "add", "kv_append", "view", "noop", "cat_sink",
})


def _env(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def probe_bandwidth(device: str, size_mb: int = 32) -> float:
    """Measure effective memory bandwidth (bytes/s) with a big copy.

    A one-time startup micro-benchmark on the current device: copies two
    ``size_mb`` MiB tensors ``a``/``b`` into ``c`` for a few iterations and
    returns ``3 * nbytes / median_time`` — the read-2/write-1 traffic of a
    typical elementwise op. On CPU this is real DRAM bandwidth; on CUDA it is
    the achievable HBM bandwidth for a memcpy-style kernel.
    """
    nbytes = size_mb * 2**20
    a = torch.empty(nbytes // 4, dtype=torch.float32, device=device)
    b = torch.empty_like(a)
    c = torch.empty_like(a)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    # Warmup
    for _ in range(3):
        torch.add(a, b, out=c)
    times = []
    for _ in range(7):
        if device.startswith("cuda"):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            torch.add(a, b, out=c)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1e-3)
        else:
            import time as _t
            t0 = _t.perf_counter()
            torch.add(a, b, out=c)
            times.append(_t.perf_counter() - t0)
    times.sort()
    return 3 * nbytes / times[len(times) // 2]


def peak_ops(device: str) -> float:
    """Compute peak (or measured) FLOP/s for the device (fallback + env)."""
    env = _env("METIS_PEAK_OPS", 0.0)
    if env:
        return env * 1e9
    kind = "cuda" if device.startswith("cuda") else "cpu"
    return _DEFAULT_PEAK_OPS[kind]


def bandwidth(device: str) -> float:
    """Effective bandwidth (B/s) for the device — probed, fallback, or env."""
    env = _env("METIS_BANDWIDTH", 0.0)
    if env:
        return env * 1e9
    try:
        return probe_bandwidth(device)
    except Exception:
        kind = "cuda" if device.startswith("cuda") else "cpu"
        return _DEFAULT_BANDWIDTH[kind]


def estimate_costs(graph, device: str) -> None:
    """Fill ``node.est_ms`` for every node using the roofline model."""
    ops = peak_ops(device)
    bw = bandwidth(device)
    for node in graph.nodes.values():
        if node.kind in _MEMORY_BOUND or node.flops == 0:
            t = node.bytes / bw if node.bytes else 0.0
        else:
            t = max(node.flops / ops, node.bytes / bw)
        node.est_ms = t * 1e3


def calibrate(graph, model, config, *, device: str, iters: int = 5) -> float:
    """Return a global scale factor ``measured_ms / roofline_ms``.

    Runs the real model forward a few times on a random batch and compares the
    measured wall time to the current roofline total. ``planner.py`` applies
    the factor to every ``est_ms`` so the plan tracks the *measured* machine
    scale (which the pure roofline cannot capture — kernel efficiency,
    framework overhead, cache effects).
    """
    import time as _t

    B, T = graph.ref_shape
    x = torch.randint(0, config.vocab_size, (B, T), device=device)
    samples = []
    with torch.no_grad():
        for _ in range(iters):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = _t.perf_counter()
            model(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples.append((_t.perf_counter() - t0) * 1e3)
    samples.sort()
    measured = samples[len(samples) // 2]
    roofline = sum(n.est_ms for n in graph.nodes.values())
    if roofline <= 0:
        return 1.0
    return max(measured / roofline, 1e-3)
