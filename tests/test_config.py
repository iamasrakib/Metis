"""
Μῆτις (Metis) — Unit Tests for Configuration System
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metis.config import ModelConfig, PRESETS, setup_logging


class TestModelConfig:
    def test_default_config(self):
        config = ModelConfig()
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_layers == 4
        assert config.max_seq_len == 256
        assert config.dropout == 0.1

    def test_from_preset_tiny(self):
        config = ModelConfig.from_preset("tiny")
        assert config.d_model == 128
        assert config.n_heads == 4
        assert config.n_layers == 4
        assert config.max_seq_len == 256

    def test_from_preset_small(self):
        config = ModelConfig.from_preset("small")
        assert config.d_model == 256
        assert config.n_heads == 4

    def test_from_preset_medium(self):
        config = ModelConfig.from_preset("medium")
        assert config.d_model == 384
        assert config.n_heads == 6
        assert config.n_layers == 6
        assert config.max_seq_len == 512

    def test_from_preset_large(self):
        config = ModelConfig.from_preset("large")
        assert config.d_model == 512
        assert config.n_heads == 8
        assert config.n_layers == 8

    def test_from_preset_with_overrides(self):
        config = ModelConfig.from_preset("tiny", max_iters=10000, learning_rate=1e-3)
        assert config.max_iters == 10000
        assert config.learning_rate == 1e-3

    def test_invalid_preset(self):
        with pytest.raises(ValueError):
            ModelConfig.from_preset("nonexistent")

    def test_gqa_defaults_to_mha(self):
        """n_kv_heads=0 should default to n_heads (MHA)."""
        config = ModelConfig(d_model=256, n_heads=8, n_kv_heads=0)
        assert config.n_kv_heads == 8  # After __post_init__
        assert config.n_groups == 1

    def test_gqa_with_kv_heads(self):
        config = ModelConfig(d_model=256, n_heads=8, n_kv_heads=4)
        assert config.n_groups == 2

    def test_gqa_invalid_heads(self):
        with pytest.raises(ValueError):
            ModelConfig(d_model=256, n_heads=8, n_kv_heads=3)  # 8 % 3 != 0

    def test_d_model_divisible(self):
        with pytest.raises(ValueError):
            ModelConfig(d_model=100, n_heads=8)  # 100 % 8 != 0

    def test_dropout_range(self):
        with pytest.raises(ValueError):
            ModelConfig(dropout=-0.1)
        with pytest.raises(ValueError):
            ModelConfig(dropout=1.5)

    def test_train_split_range(self):
        with pytest.raises(ValueError):
            ModelConfig(train_split=0.0)
        with pytest.raises(ValueError):
            ModelConfig(train_split=1.0)

    def test_max_grad_norm_positive(self):
        with pytest.raises(ValueError):
            ModelConfig(max_grad_norm=0.0)

    def test_save_json(self):
        config = ModelConfig.from_preset("tiny")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
            config.save_json(path)
        loaded = ModelConfig.from_json(path)
        assert loaded.d_model == config.d_model
        assert loaded.n_heads == config.n_heads
        assert loaded.n_layers == config.n_layers
        os.unlink(path)

    def test_from_json_partial(self):
        """from_json should tolerate extra keys in JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"d_model": 128, "n_heads": 4, "unknown_key": True}, f)
            path = f.name
        config = ModelConfig.from_json(path)
        assert config.d_model == 128
        assert config.n_heads == 4
        os.unlink(path)

    def test_summary_output(self):
        config = ModelConfig.from_preset("tiny")
        summary = config.summary()
        assert "Μῆτις" in summary
        assert "128" in summary  # d_model
        assert "tiny" in summary.lower() or "config" in summary.lower()

    def test_post_init_creates_dirs(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            config = ModelConfig(
                data_dir=os.path.join(tmp, "data"),
                checkpoint_dir=os.path.join(tmp, "checkpoints"),
                log_dir=os.path.join(tmp, "logs"),
            )
            assert os.path.isdir(config.data_dir)
            assert os.path.isdir(config.checkpoint_dir)
            assert os.path.isdir(config.log_dir)

    def test_effective_batch_size(self):
        config = ModelConfig(micro_batch_size=8, gradient_accumulation_steps=4)
        assert config.effective_batch_size == 32

    def test_head_dim(self):
        config = ModelConfig(d_model=256, n_heads=8)
        assert config.head_dim == 32

    def test_all_presets_available(self):
        assert "tiny" in PRESETS
        assert "small" in PRESETS
        assert "medium" in PRESETS
        assert "large" in PRESETS

    def test_new_fields_defaults(self):
        """Verify all new Phase 1-4 fields have sensible defaults."""
        config = ModelConfig()
        assert config.tokenizer == "char"
        assert config.use_mmap is True
        assert config.num_workers == 0
        assert config.use_moe is False
        assert config.use_qk_norm is False
        assert config.use_attention_sink is False
        assert config.use_ema is False
        assert config.use_wandb is False
        assert config.quantize == "none"
        assert config.use_flash_attn is True
        assert config.attn_backend == "auto"

    def test_attn_backend_invalid(self):
        with pytest.raises(ValueError):
            ModelConfig(attn_backend="bogus")


class TestSetupLogging:
    def test_basic_logging(self):
        logger = setup_logging("INFO")
        assert logger is not None
        assert logger.level == 20  # INFO
