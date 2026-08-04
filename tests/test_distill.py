"""
Μῆτις (Metis) — Tests for the infinite distillation loop (metis/distill.py)

Uses the offline MockTeacher (``mock=True``) and a tiny CPU model — never
touches the network or an API key. Verifies the loop mechanics that matter:
exact step counting, artifact writing, the STOP file, and (most importantly)
that a restart resumes from the last step WITHOUT re-fitting the tokenizer.
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig
from metis.distill import DistillOptions, distill
from metis.teacher import TeacherError, build_teacher


def _config(ckpt_dir: str) -> ModelConfig:
    return ModelConfig(
        dataset_path="",
        data_dir=os.path.join(ckpt_dir, "data"),
        log_dir=os.path.join(ckpt_dir, "logs"),
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=32,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_iters=10_000,
        max_grad_norm=1.0,
        warmup_steps=2,
        learning_rate=1e-3,
        device="cpu",
        tokenizer="char",
        checkpoint_dir=ckpt_dir,
        log_level="WARNING",
        use_pipeline=False,
        use_cuda_graphs=False,
        use_packing=False,
    )


def _opts(**overrides) -> DistillOptions:
    base = dict(
        max_steps=3,
        save_every=1,
        steps_per_call=4,
        max_tokens=512,
        min_sleep=0.0,
        mock=True,
    )
    base.update(overrides)
    return DistillOptions(**base)


def _state(ckpt_dir: str) -> dict:
    with open(os.path.join(ckpt_dir, "distill_state.json")) as f:
        return json.load(f)


class TestFreshRun:
    def test_writes_artifacts_and_runs_exact_steps(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        rc = distill(_config(ckpt), _opts(max_steps=3))

        assert rc == 0
        for name in ("latest_checkpoint.pt", "distill_state.json",
                     "tokenizer.json", "config.json"):
            assert os.path.exists(os.path.join(ckpt, name)), name
        state = _state(ckpt)
        assert state["step"] == 3
        assert state["tokens_seen"] > 0
        assert state["api_calls"] > 0

    def test_stop_file_stops_before_any_work(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "STOP"), "w") as f:
            f.write("")
        rc = distill(_config(ckpt), _opts(max_steps=0))

        assert rc == 0
        assert not os.path.exists(os.path.join(ckpt, "latest_checkpoint.pt"))


class TestResume:
    def test_resume_continues_and_does_not_refit_tokenizer(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        distill(_config(ckpt), _opts(max_steps=3))

        tok_path = os.path.join(ckpt, "tokenizer.json")
        mtime_before = os.path.getmtime(tok_path)
        time.sleep(0.02)

        distill(_config(ckpt), _opts(max_steps=6))

        state = _state(ckpt)
        assert state["step"] == 6
        # tokenizer.json is loaded on resume, never re-fit (mtime unchanged).
        assert os.path.getmtime(tok_path) == mtime_before

    def test_no_resume_starts_fresh(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        distill(_config(ckpt), _opts(max_steps=2))
        distill(_config(ckpt), _opts(max_steps=2, no_resume=True))

        assert _state(ckpt)["step"] == 2  # fresh run, not continued to 4


class TestTeacherResolution:
    def test_build_teacher_requires_key(self, monkeypatch):
        from types import SimpleNamespace
        for env in ("METIS_TEACHER_BASE_URL", "METIS_TEACHER_API_KEY",
                    "METIS_TEACHER_MODEL"):
            monkeypatch.delenv(env, raising=False)
        with pytest.raises(TeacherError):
            build_teacher(SimpleNamespace(teacher_base_url=None,
                                          teacher_api_key=None,
                                          teacher_model=None))
