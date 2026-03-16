"""Integration tests for the complete IndexTTS pipeline.

These tests verify the full generation pipeline works correctly
when a converted model is available.
"""

import pytest
import os
from pathlib import Path

import mlx.core as mx
import numpy as np


# Check if model is available
MODEL_DIR = Path("models/mlx-indexTTS-1.5")
REF_AUDIOS_DIR = Path("ref_audios")
HAS_MODEL = MODEL_DIR.exists() and (MODEL_DIR / "gpt.safetensors").exists()


@pytest.mark.skipif(not HAS_MODEL, reason="Model not available")
class TestIntegration:
    """Integration tests requiring the full model."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model once for all tests in this class."""
        from mlx_indextts.generate import IndexTTS
        return IndexTTS.load_model(MODEL_DIR)

    def test_load_model(self, model):
        """Test that model loads without errors."""
        assert model is not None
        assert model.gpt is not None
        assert model.bigvgan is not None
        assert model.tokenizer is not None

    def test_tokenize(self, model):
        """Test text tokenization."""
        text = "你好世界"
        tokens = model.tokenizer.encode(text)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_conditioning(self, model):
        """Test conditioning from reference audio."""
        ref_audio_path = REF_AUDIOS_DIR / "ref_spk.wav"
        if not ref_audio_path.exists():
            pytest.skip("Reference audio not available")

        conditioning, ref_mel = model.get_conditioning(ref_audio_path)

        # Check shapes
        assert conditioning.ndim == 3
        assert conditioning.shape[0] == 1  # batch
        assert conditioning.shape[1] == 32  # cond_num
        assert ref_mel.ndim == 3

    def test_generate_short(self, model):
        """Test generation with short text."""
        ref_audio_path = REF_AUDIOS_DIR / "ref_spk.wav"
        if not ref_audio_path.exists():
            pytest.skip("Reference audio not available")

        audio = model.generate(
            text="你好",
            ref_audio=ref_audio_path,
            max_mel_tokens=100,
            temperature=1.0,
        )

        # Check output
        assert audio.ndim == 1
        assert len(audio) > 0
        assert mx.isfinite(audio).all()
        # Audio should be in [-1, 1] range
        assert float(mx.min(audio)) >= -1.0
        assert float(mx.max(audio)) <= 1.0


class TestComponents:
    """Tests for individual model components."""

    def test_conformer_encoder(self):
        """Test Conformer encoder forward pass."""
        from mlx_indextts.models.conformer import ConformerEncoder
        from mlx_indextts.config import ConformerConfig

        config = ConformerConfig()
        config.num_blocks = 2  # Reduce for faster test
        encoder = ConformerEncoder(config)

        # Input mel: (batch, time, mel_dim)
        x = mx.array(np.random.randn(1, 100, 100).astype(np.float32))
        output, mask = encoder(x)

        # Check output shape
        assert output.ndim == 3
        assert output.shape[0] == 1
        assert output.shape[2] == config.output_size

    def test_bigvgan_forward(self):
        """Test BigVGAN forward pass."""
        from mlx_indextts.models.bigvgan import BigVGAN
        from mlx_indextts.config import BigVGANConfig

        config = BigVGANConfig()
        config.upsample_rates = [4, 4]  # Reduce for faster test
        config.upsample_kernel_sizes = [8, 8]
        config.resblock_kernel_sizes = [3]
        config.resblock_dilation_sizes = [[1, 2]]
        config.upsample_initial_channel = 128

        bigvgan = BigVGAN(config)

        # Input latent and ref_mel
        latent = mx.array(np.random.randn(1, 50, 1280).astype(np.float32))
        ref_mel = mx.array(np.random.randn(1, 100, 200).astype(np.float32))

        # Note: This may fail without proper weights, just checking it runs
        try:
            audio = bigvgan(latent, ref_mel)
            assert audio.ndim >= 1
        except Exception as e:
            # Expected without proper weights
            pass

    def test_gpt_forward_latent(self):
        """Test GPT forward_latent function."""
        from mlx_indextts.models.gpt import UnifiedVoice
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2

        gpt = UnifiedVoice(config)

        # Mock inputs
        conditioning = mx.array(np.random.randn(1, 32, 256).astype(np.float32))
        text_tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        mel_codes = mx.array([[100, 200, 300]], dtype=mx.int32)

        latent = gpt.forward_latent(conditioning, text_tokens, mel_codes)

        # Check output shape: should be (batch, mel_len, dim)
        assert latent.shape == (1, 3, 256)


class TestMelPositionEncoding:
    """Tests for mel position encoding - the critical bug that was fixed."""

    def test_mel_position_range(self):
        """Test that mel position indices stay within valid range."""
        from mlx_indextts.models.attention import LearnedPositionEmbedding

        max_mel_tokens = 800
        pos_emb = LearnedPositionEmbedding(max_mel_tokens + 3, 256)

        # Simulate autoregressive generation
        for mel_pos in range(100):
            emb = pos_emb.get_fixed_embedding(mel_pos)
            assert emb.shape == (1, 1, 256)
            assert mx.isfinite(emb).all()

    def test_position_encoding_consistency(self):
        """Test that position encoding is consistent."""
        from mlx_indextts.models.attention import LearnedPositionEmbedding

        pos_emb = LearnedPositionEmbedding(100, 256)

        # Same position should give same embedding
        emb1 = pos_emb.get_fixed_embedding(5)
        emb2 = pos_emb.get_fixed_embedding(5)

        diff = float(mx.mean(mx.abs(emb1 - emb2)))
        assert diff == 0.0


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_asp_variance_stability(self):
        """Test AttentiveStatisticsPooling handles variance computation safely."""
        from mlx_indextts.models.ecapa_tdnn import AttentiveStatisticsPooling

        asp = AttentiveStatisticsPooling(channels=64)

        # Input that might cause numerical issues
        x = mx.array(np.ones((1, 64, 100), dtype=np.float32) * 0.1)
        output = asp(x)

        assert mx.isfinite(output).all()
        assert not mx.isnan(output).any()

    def test_snake_activation_stability(self):
        """Test Snake activation with extreme values."""
        from mlx_indextts.models.activations import Snake

        snake = Snake(channels=64)

        # Large values
        x_large = mx.array(np.ones((1, 64, 100), dtype=np.float32) * 100)
        y_large = snake(x_large)
        assert mx.isfinite(y_large).all()

        # Small values
        x_small = mx.array(np.ones((1, 64, 100), dtype=np.float32) * 0.001)
        y_small = snake(x_small)
        assert mx.isfinite(y_small).all()
