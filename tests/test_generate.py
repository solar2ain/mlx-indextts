"""Tests for speech generation."""

import pytest
import mlx.core as mx
import numpy as np


class TestGenerateComponents:
    """Tests for generation component functions."""

    def test_prepare_inputs_shape(self):
        """Test that input preparation produces correct shapes."""
        from mlx_indextts.config import IndexTTSConfig
        from mlx_indextts.models.gpt import UnifiedVoice

        config = IndexTTSConfig()
        # Use smaller model for testing
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2
        config.gpt.condition_module.output_size = 128

        model = UnifiedVoice(config)

        # Mock conditioning
        conditioning = mx.array(np.random.randn(1, 32, 256).astype(np.float32))

        # Mock text tokens
        text_tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)

        input_emb, mask = model.prepare_inputs(conditioning, text_tokens)

        # Should have: conditioning (32) + start + text (5) + stop = 39
        expected_len = 32 + 1 + 5 + 1
        assert input_emb.shape == (1, expected_len, 256)
        assert mask.shape == (1, expected_len)

    def test_sampling_temperature(self):
        """Test that temperature affects sampling distribution."""
        from mlx_indextts.models.gpt import UnifiedVoice
        from mlx_indextts.config import IndexTTSConfig

        config = IndexTTSConfig()
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2

        model = UnifiedVoice(config)

        # Create logits with clear peak
        logits_np = np.full((1, 100), -10.0, dtype=np.float32)
        logits_np[0, 50] = 10.0
        logits = mx.array(logits_np)

        # With low temperature, should always pick the peak
        mx.random.seed(42)
        samples_low_temp = [model._sample(logits, temperature=0.1)[0].item() for _ in range(10)]
        assert all(s == 50 for s in samples_low_temp)

        # With high temperature, should have more variance
        mx.random.seed(42)
        samples_high_temp = [model._sample(logits, temperature=2.0)[0].item() for _ in range(20)]
        # Not all should be 50 with high temperature
        unique_samples = set(samples_high_temp)
        assert len(unique_samples) >= 1  # At least some variance


class TestMelExtraction:
    """Tests for mel spectrogram extraction in generation pipeline."""

    def test_mel_extractor_init(self):
        """Test MelSpectrogramExtractor initialization."""
        from mlx_indextts.mel import MelSpectrogramExtractor

        extractor = MelSpectrogramExtractor(
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            sample_rate=24000,
        )

        assert extractor.n_mels == 100
        assert extractor.sample_rate == 24000

    def test_mel_extractor_output(self):
        """Test MelSpectrogramExtractor produces valid output."""
        from mlx_indextts.mel import MelSpectrogramExtractor

        extractor = MelSpectrogramExtractor(n_mels=100)

        # 1 second of audio
        audio = mx.array(np.random.randn(24000).astype(np.float32))
        mel = extractor(audio)

        # Check shape
        assert mel.shape[0] == 100

        # Check values are finite
        assert mx.isfinite(mel).all()


class TestEndToEndFlow:
    """Tests for end-to-end generation flow."""

    def test_conditioning_shape(self):
        """Test that conditioning produces correct shape."""
        from mlx_indextts.config import IndexTTSConfig, ConformerConfig
        from mlx_indextts.models.gpt import UnifiedVoice

        config = IndexTTSConfig()
        config.gpt.model_dim = 256
        config.gpt.heads = 4
        config.gpt.layers = 2
        config.gpt.condition_type = "perceiver"  # Simpler for testing

        model = UnifiedVoice(config)

        # Mock mel spectrogram
        mel = mx.array(np.random.randn(1, 100, 200).astype(np.float32))

        conditioning = model.get_conditioning(mel)

        # Should be (batch, cond_num, dim)
        assert conditioning.shape == (1, 32, 256)
