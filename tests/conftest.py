"""Pytest configuration and fixtures."""

import pytest
import mlx.core as mx
import numpy as np


@pytest.fixture
def sample_audio():
    """Generate sample audio data."""
    # 1 second of audio at 24kHz
    return mx.array(np.random.randn(24000).astype(np.float32))


@pytest.fixture
def sample_mel():
    """Generate sample mel spectrogram."""
    # (batch, n_mels, time)
    return mx.array(np.random.randn(1, 100, 200).astype(np.float32))


@pytest.fixture
def sample_text_tokens():
    """Generate sample text tokens."""
    return mx.array([[100, 200, 300, 400, 500]], dtype=mx.int32)


@pytest.fixture
def small_config():
    """Create a small config for testing."""
    from mlx_indextts.config import IndexTTSConfig, GPTConfig, ConformerConfig

    config = IndexTTSConfig()
    config.gpt.model_dim = 256
    config.gpt.heads = 4
    config.gpt.layers = 2
    config.gpt.max_mel_tokens = 100
    config.gpt.max_text_tokens = 50
    config.gpt.condition_module = ConformerConfig(
        output_size=128,
        attention_heads=4,
        num_blocks=2,
    )
    config.bigvgan.gpt_dim = 256
    config.bigvgan.upsample_initial_channel = 256

    return config
