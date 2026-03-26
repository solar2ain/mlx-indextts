"""Tests for model components."""

import pytest
import mlx.core as mx
import numpy as np


class TestMelSpectrogram:
    """Tests for mel spectrogram extraction."""

    def test_mel_shape(self):
        """Test that mel spectrogram has correct shape."""
        from mlx_indextts.mel import log_mel_spectrogram

        # Create dummy audio (1 second at 24kHz)
        audio = mx.array(np.random.randn(24000).astype(np.float32))

        mel = log_mel_spectrogram(audio, n_mels=100, hop_length=256)

        # Should have 100 mel bins
        assert mel.shape[0] == 100
        # Approximately 24000 / 256 = 93.75 frames
        assert mel.shape[1] > 90

    def test_mel_batch(self):
        """Test mel spectrogram with batch dimension."""
        from mlx_indextts.mel import log_mel_spectrogram

        # Batch of 2 audio samples
        audio = mx.array(np.random.randn(2, 24000).astype(np.float32))

        mel = log_mel_spectrogram(audio, n_mels=100)

        assert mel.shape[0] == 2
        assert mel.shape[1] == 100


class TestActivations:
    """Tests for activation functions."""

    def test_snake_shape(self):
        """Test Snake activation preserves shape."""
        from mlx_indextts.models.activations import Snake

        snake = Snake(channels=64)
        x = mx.array(np.random.randn(2, 64, 100).astype(np.float32))

        y = snake(x)

        assert y.shape == x.shape

    def test_snakebeta_shape(self):
        """Test SnakeBeta activation preserves shape."""
        from mlx_indextts.models.activations import SnakeBeta

        snake = SnakeBeta(channels=64)
        x = mx.array(np.random.randn(2, 64, 100).astype(np.float32))

        y = snake(x)

        assert y.shape == x.shape


class TestAttention:
    """Tests for attention modules."""

    def test_multihead_attention(self):
        """Test multi-head attention output shape."""
        from mlx_indextts.models.attention import MultiHeadAttention

        attn = MultiHeadAttention(dim=512, num_heads=8)
        x = mx.array(np.random.randn(2, 32, 512).astype(np.float32))

        y, cache = attn(x, x, x)

        assert y.shape == x.shape
        assert cache is not None

    def test_learned_position_embedding(self):
        """Test learned position embeddings."""
        from mlx_indextts.models.attention import LearnedPositionEmbedding

        pos_emb = LearnedPositionEmbedding(max_seq_len=512, dim=256)
        x = mx.array(np.random.randn(2, 32, 256).astype(np.float32))

        pe = pos_emb(x)

        assert pe.shape == (32, 256)


class TestPerceiver:
    """Tests for Perceiver Resampler."""

    def test_perceiver_shape(self):
        """Test Perceiver output shape."""
        from mlx_indextts.models.perceiver import PerceiverResampler

        perceiver = PerceiverResampler(
            dim=512,
            n_dim_context=256,
            n_latents=32,
        )

        # Context from encoder
        context = mx.array(np.random.randn(2, 100, 256).astype(np.float32))

        latents = perceiver(context)

        assert latents.shape == (2, 32, 512)


class TestGPT2:
    """Tests for GPT-2 model."""

    def test_gpt2_forward(self):
        """Test GPT-2 forward pass."""
        from mlx_indextts.models.gpt2 import GPT2Model

        gpt = GPT2Model(dim=256, num_heads=4, num_layers=2)

        x = mx.array(np.random.randn(2, 32, 256).astype(np.float32))
        y, cache = gpt(x)

        assert y.shape == x.shape
        assert len(cache) == 2  # One per layer

    def test_gpt2_with_cache(self):
        """Test GPT-2 with KV cache for incremental decoding."""
        from mlx_indextts.models.gpt2 import GPT2Model

        gpt = GPT2Model(dim=256, num_heads=4, num_layers=2)

        # First pass
        x = mx.array(np.random.randn(2, 32, 256).astype(np.float32))
        _, cache = gpt(x)

        # Incremental pass
        x_new = mx.array(np.random.randn(2, 1, 256).astype(np.float32))
        y, new_cache = gpt(x_new, cache=cache)

        assert y.shape == (2, 1, 256)
        # Cache should have grown
        assert new_cache[0][0].shape[1] == 33


class TestECAPATDNN:
    """Tests for ECAPA-TDNN speaker encoder."""

    def test_ecapa_output_shape(self):
        """Test ECAPA-TDNN output shape."""
        from mlx_indextts.models.ecapa_tdnn import ECAPATDNN

        ecapa = ECAPATDNN(input_size=100, lin_neurons=512)

        # Input mel spectrogram
        mel = mx.array(np.random.randn(2, 100, 200).astype(np.float32))

        emb = ecapa(mel)

        # Output should be (batch, 1, lin_neurons)
        assert emb.shape == (2, 1, 512)


class TestTokenizer:
    """Tests for text tokenizer."""

    def test_normalize(self):
        """Test text normalization."""
        from mlx_indextts.normalize import normalize_text, tokenize_by_cjk_char

        # Basic normalization
        text = "Hello  World\n\tTest"
        normalized = normalize_text(text)
        assert normalized == "Hello World Test"

        # CJK tokenization
        text = "Hello你好World"
        tokenized = tokenize_by_cjk_char(text)
        assert " 你 " in tokenized
        assert " 好 " in tokenized

    def test_remove_emoji(self):
        """Test emoji removal from text (replaced with space)."""
        from mlx_indextts.normalize import remove_emoji

        # Common emojis - replaced with space
        assert remove_emoji("你好😀世界") == "你好 世界"
        assert remove_emoji("开心😊") == "开心 "
        assert remove_emoji("Hello 👋 World") == "Hello   World"

        # Multiple emojis
        assert remove_emoji("测试🎉🎊🎁") == "测试   "

        # No emoji
        assert remove_emoji("正常文本") == "正常文本"
        assert remove_emoji("Hello World") == "Hello World"

        # Emoji at end
        assert remove_emoji("吃饭了吗？🍲") == "吃饭了吗？ "
