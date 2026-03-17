"""Tests for IndexTTS 2.0 specific model components."""

import pytest
import mlx.core as mx
import numpy as np
from pathlib import Path


class TestBigVGANV2:
    """Tests for BigVGAN v2 vocoder."""

    def test_bigvgan_v2_config(self):
        """Test BigVGAN v2 config defaults."""
        from mlx_indextts.models.bigvgan_v2 import BigVGANV2Config

        config = BigVGANV2Config()
        assert config.num_mels == 80
        assert config.upsample_initial_channel == 1536
        assert len(config.upsample_rates) == 6
        assert config.activation == "snakebeta"
        assert config.use_tanh_at_final is False

    def test_bigvgan_v2_forward_shape(self):
        """Test BigVGAN v2 forward pass output shape."""
        from mlx_indextts.models.bigvgan_v2 import BigVGANV2, BigVGANV2Config

        # Use minimal config for faster test
        config = BigVGANV2Config(
            num_mels=80,
            upsample_rates=[4, 4],
            upsample_kernel_sizes=[8, 8],
            upsample_initial_channel=128,
            resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1, 2]],
        )
        model = BigVGANV2(config)

        # Input mel: (batch, num_mels, time)
        mel = mx.array(np.random.randn(1, 80, 50).astype(np.float32))
        audio = model(mel)

        # Output: (batch, 1, time * prod(upsample_rates))
        # 50 * 4 * 4 = 800
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] == 50 * 4 * 4


class TestDiT:
    """Tests for Diffusion Transformer (DiT) components."""

    def test_timestep_embedder(self):
        """Test timestep embedding generation."""
        from mlx_indextts.models.s2mel.dit import TimestepEmbedder

        embedder = TimestepEmbedder(hidden_size=256)

        # Test single timestep
        t = mx.array([0.5])
        emb = embedder(t)
        assert emb.shape == (1, 256)
        assert mx.isfinite(emb).all()

        # Test batch
        t_batch = mx.array([0.0, 0.25, 0.5, 0.75, 1.0])
        emb_batch = embedder(t_batch)
        assert emb_batch.shape == (5, 256)

    def test_model_args_defaults(self):
        """Test ModelArgs configuration defaults."""
        from mlx_indextts.models.s2mel.dit import ModelArgs

        args = ModelArgs()
        assert args.n_layer == 13
        assert args.n_head == 8
        assert args.dim == 512
        assert args.head_dim == 64  # dim // n_head
        assert args.intermediate_size is not None

    def test_rotary_position_embedding(self):
        """Test RotaryPositionEmbedding class."""
        from mlx_indextts.models.s2mel.dit import RotaryPositionEmbedding

        rope = RotaryPositionEmbedding(dim=64, max_seq_len=512)

        # Check freqs_cis is computed
        assert hasattr(rope, 'freqs_cis')
        assert rope.freqs_cis.shape[0] == 512  # max_seq_len

    def test_rmsnorm(self):
        """Test RMSNorm layer."""
        from mlx_indextts.models.s2mel.dit import RMSNorm

        norm = RMSNorm(dims=256)  # Note: parameter is 'dims' not 'dim'
        x = mx.array(np.random.randn(2, 32, 256).astype(np.float32))
        y = norm(x)

        assert y.shape == x.shape
        assert mx.isfinite(y).all()


class TestLengthRegulator:
    """Tests for Length Regulator (InterpolateRegulator)."""

    def test_length_regulator_forward(self):
        """Test length regulator forward pass."""
        from mlx_indextts.models.s2mel.length_regulator import InterpolateRegulator

        lr = InterpolateRegulator(
            channels=256,
            sampling_ratios=(1, 1),
            is_discrete=False,
            in_channels=512,
        )

        # Input: (batch, seq_len, in_channels)
        x = mx.array(np.random.randn(1, 20, 512).astype(np.float32))
        ylens = mx.array([50])  # Target length

        result = lr(x, ylens)

        # Returns tuple: (output, output_lengths, None, None, None)
        assert isinstance(result, tuple)
        out = result[0]

        # Output is (batch, target_len, channels) - NLC format
        assert out.shape == (1, 50, 256)

    def test_length_regulator_interpolation(self):
        """Test that interpolation produces correct length."""
        from mlx_indextts.models.s2mel.length_regulator import InterpolateRegulator

        lr = InterpolateRegulator(
            channels=128,
            sampling_ratios=(1,),
            is_discrete=False,
            in_channels=256,
        )

        x = mx.array(np.random.randn(1, 10, 256).astype(np.float32))

        # Test various target lengths
        for target_len in [20, 50, 100]:
            ylens = mx.array([target_len])
            out, _, _, _, _ = lr(x, ylens)
            # Output is (batch, target_len, channels) - NLC format
            assert out.shape == (1, target_len, 128)


class TestWaveNet:
    """Tests for WaveNet components."""

    def test_sconv1d(self):
        """Test SConv1d module."""
        from mlx_indextts.models.s2mel.wavenet import SConv1d

        conv = SConv1d(in_channels=64, out_channels=128, kernel_size=3)

        # Input: NCL format
        x = mx.array(np.random.randn(1, 64, 50).astype(np.float32))
        y = conv(x)

        assert y.shape == (1, 128, 50)

    def test_wavenet_forward(self):
        """Test WaveNet module forward pass."""
        from mlx_indextts.models.s2mel.wavenet import WN

        hidden_channels = 64
        gin_channels = 128

        wn = WN(
            hidden_channels=hidden_channels,
            kernel_size=3,
            dilation_rate=1,
            n_layers=2,
            gin_channels=gin_channels,
        )
        wn.eval()  # Critical: disable dropout

        # Input x: (batch, hidden, length)
        x = mx.array(np.random.randn(1, hidden_channels, 50).astype(np.float32))
        # Conditioning g: (batch, gin, 1) - will be broadcast
        g = mx.array(np.random.randn(1, gin_channels, 1).astype(np.float32))
        # Mask: same shape as x
        x_mask = mx.ones((1, 1, 50))

        out = wn(x, x_mask=x_mask, g=g)

        assert out.shape == x.shape
        assert mx.isfinite(out).all()


class TestCFM:
    """Tests for Conditional Flow Matching."""

    def test_cfm_module_exists(self):
        """Test CFM module can be imported."""
        from mlx_indextts.models.s2mel.cfm import CFM

        # CFM requires full DiT setup, just check import works
        assert CFM is not None

    def test_create_cfm_from_config(self):
        """Test CFM factory function."""
        from mlx_indextts.models.s2mel.cfm import create_cfm_from_config

        # Check function exists
        assert create_cfm_from_config is not None


class TestGPTV2:
    """Tests for GPT v2 model (with emotion support)."""

    def test_gpt_v2_conditioning_encoder(self):
        """Test GPT v2 ConditioningEncoder."""
        from mlx_indextts.models.gpt_v2 import ConditioningEncoder

        # Correct signature: spec_dim, embedding_dim
        encoder = ConditioningEncoder(
            spec_dim=1024,
            embedding_dim=256,
        )

        # Input: (batch, spec_dim, time) - NCL format
        x = mx.array(np.random.randn(1, 1024, 100).astype(np.float32))

        out = encoder(x)

        assert out.shape[0] == 1
        assert out.shape[1] == 256  # embedding_dim
        assert out.shape[2] == 100  # time

    def test_gpt_v2_class_exists(self):
        """Test UnifiedVoiceV2 class exists."""
        from mlx_indextts.models.gpt_v2 import UnifiedVoiceV2

        # Just check class can be imported
        assert UnifiedVoiceV2 is not None


class TestS2Mel:
    """Tests for S2Mel module."""

    def test_s2mel_import(self):
        """Test S2Mel can be imported."""
        from mlx_indextts.models.s2mel.s2mel import S2Mel

        assert S2Mel is not None


# Integration test for v2.0 model (requires model files AND torch)
MODEL_DIR_V2 = Path("models/mlx-indexTTS-2.0")
HAS_MODEL_V2 = MODEL_DIR_V2.exists() and (MODEL_DIR_V2 / "gpt.safetensors").exists()

# Check if torch is available (needed for generate_v2 imports)
try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not (HAS_MODEL_V2 and HAS_TORCH), reason="v2.0 Model or torch not available")
class TestIntegrationV2:
    """Integration tests for IndexTTS 2.0."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load v2.0 model once for all tests."""
        from mlx_indextts.generate_v2 import IndexTTSv2
        return IndexTTSv2(MODEL_DIR_V2)

    def test_load_model_v2(self, model):
        """Test that v2.0 model loads without errors."""
        assert model is not None
        assert model.gpt_mlx is not None
        assert model.s2mel_mlx is not None
        assert model.bigvgan_mlx is not None

    def test_vq2emb_mlx(self, model):
        """Test MLX vq2emb implementation."""
        # Create dummy codes
        codes = mx.array([[100, 200, 300, 400]], dtype=mx.int32)
        emb = model._vq2emb_forward(codes)

        # Output should be (batch, 1024, seq_len)
        assert emb.shape == (1, 1024, 4)
        assert mx.isfinite(emb).all()

    def test_emotion_matrix_loading(self, model):
        """Test that emotion matrices are loaded."""
        # Check emotion matrices exist (internal attributes)
        assert hasattr(model, '_spk_matrix')
        assert hasattr(model, '_emo_matrix')
