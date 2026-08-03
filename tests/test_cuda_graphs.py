"""
Μῆτις (Metis) — Unit Tests for CUDA Graphs training step
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig  # noqa: E402
from metis.cuda_graphs import CUDAGraphStep  # noqa: E402
from metis.model import MetisLM  # noqa: E402

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="CUDA required")


def make_cfg(**kw) -> ModelConfig:
    defaults = dict(
        d_model=64,
        n_heads=4,
        n_kv_heads=0,
        n_layers=2,
        max_seq_len=32,
        vocab_size=256,
        dropout=0.0,
        use_flash_attn=True,
        micro_batch_size=4,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        learning_rate=3e-4,
        use_moe=False,
        tie_weights=True,
        device="cuda" if CUDA else "cpu",
    )
    defaults.update(kw)
    return ModelConfig(**defaults)


AMP = torch.bfloat16 if (CUDA and torch.cuda.is_bf16_supported()) else torch.float16


def _rand_batches(cfg, seed=0):
    torch.manual_seed(seed)
    B, T = cfg.micro_batch_size, cfg.max_seq_len
    return [
        (torch.randint(0, cfg.vocab_size, (B, T)),
         torch.randint(0, cfg.vocab_size, (B, T)))
        for _ in range(cfg.gradient_accumulation_steps)
    ]


def _build(cfg):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = MetisLM(cfg).to(cfg.device).train()
    opt = model.configure_optimizers(0.1, cfg.learning_rate, cfg.device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return model, opt, scaler


class TestCapabilityFallback:
    """Configs that cannot be captured must report inactive, not raise."""

    @pytest.mark.parametrize("kw,reason_frag", [
        ({"use_moe": True, "moe_num_experts": 4, "moe_top_k": 2}, "data-dependent"),
        ({"use_ddp": True}, "DDP"),
        ({"compile_model": True}, "compile"),
    ])
    def test_inactive_reason(self, kw, reason_frag):
        cfg = make_cfg(**kw)
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        assert not step.active
        # The config-specific fallback reasons ("data-dependent", "DDP",
        # "compile") are only reachable on CUDA; on a CPU-only host the
        # capability check short-circuits on the device first.
        if CUDA:
            assert reason_frag in step.reason
        else:
            assert "not a CUDA device" in step.reason

    @pytest.mark.parametrize("kw,reason_frag", [
        ({"use_moe": True, "moe_num_experts": 4, "moe_top_k": 2}, "data-dependent"),
        ({"use_ddp": True}, "DDP"),
    ])
    def test_inactive_train_step_still_works(self, kw, reason_frag):
        """Inactive graphs must still run the eager fallback correctly."""
        cfg = make_cfg(**kw)
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        assert not step.active
        for _ in range(2):
            la, gn = step.train_step(_rand_batches(cfg, 1))
            assert la == la  # not NaN
            assert gn == gn
        assert all(torch.isfinite(p.data).all() for p in model.parameters())


@gpu
class TestGraphCapture:
    def test_capture_activates(self):
        cfg = make_cfg()
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        assert step.active, step.reason
        assert step.reason == "captured"
        assert step.graph is not None
        assert len(step.static_x) == cfg.gradient_accumulation_steps

    def test_warmup_invisible(self):
        """Capture must not perturb model/optimizer/scaler state."""
        cfg = make_cfg()
        model, opt, scaler = _build(cfg)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        assert step.active
        after = model.state_dict()
        for k, v in before.items():
            assert torch.equal(v, after[k]), f"weight {k} changed by warmup"

    def test_train_step_runs_and_returns(self):
        cfg = make_cfg()
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        for _ in range(3):
            la, gn = step.train_step(_rand_batches(cfg, 7))
            assert la > 0 and la == la
            assert gn == gn
        assert step.last_loss == pytest.approx(la)

    def test_bit_identical_to_eager(self):
        """dropout=0: graph replay == eager reference (weights & losses)."""
        cfg = make_cfg(dropout=0.0)
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )

        # Reference model with identical weights + fresh seed → same RNG start.
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        ref = MetisLM(cfg).to(cfg.device).train()
        ref.load_state_dict(model.state_dict())
        ref_opt = ref.configure_optimizers(0.1, cfg.learning_rate, cfg.device)
        ref_opt.load_state_dict(opt.state_dict())
        ref_sc = torch.amp.GradScaler("cuda", enabled=True)
        ref_sc.load_state_dict(scaler.state_dict())
        rng = torch.cuda.get_rng_state()

        for i in range(4):
            batches = _rand_batches(cfg, 100 + i)
            la_g, gn_g = step.train_step(batches)
            # eager reference
            torch.cuda.set_rng_state(rng)
            N = cfg.gradient_accumulation_steps
            ref_opt.zero_grad(set_to_none=True)
            la_e = 0.0
            for x, y in batches:
                x, y = x.to(cfg.device, non_blocking=True), y.to(cfg.device, non_blocking=True)
                with torch.autocast("cuda", dtype=AMP, cache_enabled=False):
                    _, loss, _ = ref(x, y, use_checkpointing=False)
                    loss = loss / N
                ref_sc.scale(loss).backward()
                la_e += loss.item()
            ref_sc.unscale_(ref_opt)
            gn_e = float(torch.nn.utils.clip_grad_norm_(ref.parameters(), cfg.max_grad_norm))
            ref_sc.step(ref_opt)
            ref_sc.update()
            rng = torch.cuda.get_rng_state()

            assert la_g == la_e, f"loss diverged at step {i}: {la_g} vs {la_e}"
            assert gn_g == gn_e
            assert all(
                torch.equal(a.data, b.data)
                for a, b in zip(model.parameters(), ref.parameters())
            ), f"weights diverged at step {i}"

    def test_grad_addresses_stable(self):
        """zero_grad(set_to_none=False) keeps capture-time pool addresses."""
        cfg = make_cfg()
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        first = {n: p.grad.data_ptr() for n, p in model.named_parameters() if p.grad is not None}
        assert len(first) > 0
        for _ in range(3):
            step.train_step(_rand_batches(cfg, 5))
        for n, p in model.named_parameters():
            if n in first:
                assert p.grad is not None and p.grad.data_ptr() == first[n]

    def test_dropout_masks_not_frozen(self):
        """Two replays of the same inputs must consume fresh dropout masks."""
        cfg = make_cfg(dropout=0.4)
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        la1, _ = step.train_step(_rand_batches(cfg, 11))
        la2, _ = step.train_step(_rand_batches(cfg, 11))
        assert la1 != la2, "dropout mask appears frozen across replays"

    def test_scaler_overflow_backs_off(self):
        """fp16 overflow must skip the step and back the scale off."""
        cfg = make_cfg(dropout=0.0, learning_rate=1e-2, max_grad_norm=100.0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = MetisLM(cfg).to(cfg.device).train()
        opt = model.configure_optimizers(0.0, cfg.learning_rate, cfg.device)
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=2.0**24)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=torch.float16, device=cfg.device, warmup_iters=2,
        )
        assert step.active
        s0 = float(scaler.get_scale())
        x = torch.randint(0, cfg.vocab_size, (cfg.micro_batch_size, cfg.max_seq_len))
        y = torch.randint(0, cfg.vocab_size, (cfg.micro_batch_size, cfg.max_seq_len))
        # force high-magnitude grads under fp16
        for _ in range(3):
            step.train_step([(x, y)] * cfg.gradient_accumulation_steps)
        s1 = float(scaler.get_scale())
        assert s1 < s0, f"scale did not back off: {s0} → {s1}"
        assert all(torch.isfinite(p.data).all() for p in model.parameters())

    def test_scaler_reset_after_warmup(self):
        """The warmup's 3 steps must not advance the scaler (fresh init_scale)."""
        cfg = make_cfg()
        model, opt, scaler = _build(cfg)
        # A fresh scaler starts at init_scale with tracker 0.
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        assert step.active
        assert scaler._scale is not None
        assert float(scaler._scale) == pytest.approx(2.0**16)
        assert float(scaler._growth_tracker) == 0.0

    def test_info_report(self):
        cfg = make_cfg()
        model, opt, scaler = _build(cfg)
        step = CUDAGraphStep(
            model, opt, scaler, cfg,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            micro_batch_size=cfg.micro_batch_size, max_seq_len=cfg.max_seq_len,
            amp_dtype=AMP, device=cfg.device,
        )
        info = step.info()
        assert info["active"] is True
        assert info["reason"] == "captured"
        assert info["gradient_accumulation_steps"] == cfg.gradient_accumulation_steps
        assert info["dtype"] == str(AMP)
