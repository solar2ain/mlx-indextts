"""Tests for model conversion."""

import pytest
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np


class TestConvertWeights:
    """Tests for weight conversion functions."""

    def test_convert_gpt_weights_transpose(self):
        """Test that GPT attention weights are transposed correctly."""
        from mlx_indextts.convert import convert_gpt_weights
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()
        config.gpt.layers = 2

        # Create mock weights with GPT-2 style attention
        weights = {
            "gpt.h.0.attn.c_attn.weight": np.random.randn(1024, 3072).astype(np.float32),
            "gpt.h.0.attn.c_proj.weight": np.random.randn(1024, 1024).astype(np.float32),
            "gpt.h.0.mlp.c_fc.weight": np.random.randn(1024, 4096).astype(np.float32),
            "gpt.h.0.mlp.c_proj.weight": np.random.randn(4096, 1024).astype(np.float32),
        }

        converted = convert_gpt_weights(weights, config)

        # Check shapes are transposed
        assert converted["gpt.h.0.attn.c_attn.weight"].shape == (3072, 1024)
        assert converted["gpt.h.0.attn.c_proj.weight"].shape == (1024, 1024)

    def test_convert_conv_weights(self):
        """Test that conv1d weights are transposed correctly."""
        from mlx_indextts.convert import convert_gpt_weights
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()

        # PyTorch Conv1d: (out_channels, in_channels, kernel_size)
        weights = {
            "conditioning_encoder.init.weight": np.random.randn(512, 100, 1).astype(np.float32),
        }

        converted = convert_gpt_weights(weights, config)

        # MLX expects (out_channels, kernel_size, in_channels)
        assert converted["conditioning_encoder.init.weight"].shape == (512, 1, 100)

    def test_convert_perceiver_kv_split(self):
        """Test that perceiver to_kv weights are split correctly."""
        from mlx_indextts.convert import convert_gpt_weights
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()

        # Perceiver to_kv contains both k and v
        weights = {
            "perceiver_encoder.layers.0.0.to_kv.weight": np.random.randn(1024, 512).astype(np.float32),
        }

        converted = convert_gpt_weights(weights, config)

        # Should be split into linear_k and linear_v
        assert "perceiver_encoder.layers.0.0.linear_k.weight" in converted
        assert "perceiver_encoder.layers.0.0.linear_v.weight" in converted
        assert converted["perceiver_encoder.layers.0.0.linear_k.weight"].shape == (512, 512)
        assert converted["perceiver_encoder.layers.0.0.linear_v.weight"].shape == (512, 512)


class TestConvertBigVGAN:
    """Tests for BigVGAN weight conversion."""

    def test_convert_upsampling_weights(self):
        """Test that upsampling transpose conv weights are handled correctly."""
        from mlx_indextts.convert import convert_bigvgan_weights
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()

        # PyTorch ConvTranspose1d: (in_channels, out_channels, kernel_size)
        weights = {
            "ups.0.0.weight": np.random.randn(1536, 768, 8).astype(np.float32),
        }

        converted = convert_bigvgan_weights(weights, config)

        # Should be transposed for MLX
        assert converted["ups.0.0.weight"].shape == (768, 8, 1536)

    def test_skip_batch_norm_tracking(self):
        """Test that batch norm tracking tensors are skipped."""
        from mlx_indextts.convert import convert_bigvgan_weights
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()

        weights = {
            "speaker_encoder.block1.norm1.running_mean": np.zeros(512).astype(np.float32),
            "speaker_encoder.block1.norm1.num_batches_tracked": np.array(0),
        }

        converted = convert_bigvgan_weights(weights, config)

        # num_batches_tracked should be skipped
        assert "num_batches_tracked" not in str(converted.keys())
