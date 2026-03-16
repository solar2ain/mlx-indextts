"""BigVGAN vocoder for IndexTTS."""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.config import BigVGANConfig
from mlx_indextts.models.activations import SnakeBeta, Snake, Activation1d
from mlx_indextts.models.ecapa_tdnn import ECAPATDNN


class AMPBlock(nn.Module):
    """Anti-aliased Multi-Periodicity block for BigVGAN.

    Note: All conv operations use NLC format (batch, length, channels).
    Activations are wrapped with Activation1d for anti-aliasing.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: List[int] = [1, 3, 5],
        activation: str = "snakebeta",
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.channels = channels

        # Convolutions with different dilations
        self.convs1 = []
        self.convs2 = []

        for d in dilations:
            padding = (kernel_size - 1) * d // 2
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding)
            )
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
            )

        # Activations for each conv (wrapped with Activation1d for anti-aliasing)
        num_convs = len(self.convs1) + len(self.convs2)

        if activation == "snakebeta":
            self.activations = [
                Activation1d(SnakeBeta(channels, alpha_logscale))
                for _ in range(num_convs)
            ]
        elif activation == "snake":
            self.activations = [
                Activation1d(Snake(channels, alpha_logscale))
                for _ in range(num_convs)
            ]
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, channels, length) - NCL format

        Returns:
            Output tensor (batch, channels, length) - NCL format
        """
        acts1 = self.activations[::2]
        acts2 = self.activations[1::2]

        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            # PyTorch order: activation -> conv -> activation -> conv
            # Activations expect NCL, Conv1d expects NLC

            # a1(x) - activation in NCL
            xt = a1(x)

            # c1(xt) - conv in NLC
            xt = xt.transpose(0, 2, 1)  # NCL -> NLC
            xt = c1(xt)
            xt = xt.transpose(0, 2, 1)  # NLC -> NCL

            # a2(xt) - activation in NCL
            xt = a2(xt)

            # c2(xt) - conv in NLC
            xt = xt.transpose(0, 2, 1)  # NCL -> NLC
            xt = c2(xt)
            xt = xt.transpose(0, 2, 1)  # NLC -> NCL

            x = xt + x

        return x


class BigVGAN(nn.Module):
    """BigVGAN vocoder with speaker conditioning.

    Converts mel latents from GPT to waveform audio.
    All operations use NLC format (batch, length, channels).
    """

    def __init__(self, config: BigVGANConfig):
        """Initialize BigVGAN.

        Args:
            config: BigVGAN configuration
        """
        super().__init__()
        self.config = config

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # Pre-convolution (NLC format)
        self.conv_pre = nn.Conv1d(
            config.gpt_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            padding=3,
        )

        # Upsampling layers
        self.ups = []
        ch = config.upsample_initial_channel
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            out_ch = ch // 2
            padding = (k - u) // 2
            self.ups.append([
                nn.ConvTranspose1d(ch, out_ch, k, stride=u, padding=padding)
            ])
            ch = out_ch

        # Residual blocks
        self.resblocks = []
        ch = config.upsample_initial_channel
        for i in range(self.num_upsamples):
            ch = ch // 2
            for j, (k, d) in enumerate(zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes
            )):
                self.resblocks.append(
                    AMPBlock(
                        ch,
                        kernel_size=k,
                        dilations=d,
                        activation=config.activation,
                        alpha_logscale=config.snake_logscale,
                    )
                )

        # Post-activation (wrapped with Activation1d for anti-aliasing)
        if config.activation == "snakebeta":
            self.activation_post = Activation1d(SnakeBeta(ch, config.snake_logscale))
        else:
            self.activation_post = Activation1d(Snake(ch, config.snake_logscale))

        # Post-convolution
        self.conv_post = nn.Conv1d(ch, 1, kernel_size=7, padding=3)

        # Speaker encoder
        self.speaker_encoder = ECAPATDNN(
            input_size=config.num_mels,
            lin_neurons=config.speaker_embedding_dim,
        )

        # Conditioning layers (NLC format)
        self.cond_layer = nn.Conv1d(
            config.speaker_embedding_dim,
            config.upsample_initial_channel,
            kernel_size=1,
        )

        if config.cond_d_vector_in_each_upsampling_layer:
            self.conds = []
            ch = config.upsample_initial_channel
            for i in range(self.num_upsamples):
                ch = ch // 2
                self.conds.append(
                    nn.Conv1d(config.speaker_embedding_dim, ch, kernel_size=1)
                )
        else:
            self.conds = []

        self.use_tanh = config.use_tanh_at_final

    def __call__(
        self,
        x: mx.array,
        mel_ref: mx.array,
    ) -> mx.array:
        """Generate audio from latents.

        Args:
            x: GPT latents (batch, seq_len, gpt_dim) - NLC format from GPT
            mel_ref: Reference mel spectrogram (batch, n_mels, time) - NCL format

        Returns:
            Generated audio (batch, 1, samples) - NCL format
        """
        # mel_ref should be (batch, n_mels, time) NCL format for speaker_encoder
        # speaker_encoder expects (batch, n_mels, time) or (batch, time, n_mels)
        # Original PyTorch: speaker_encoder receives mel_ref directly

        # Extract speaker embedding: (batch, 1, spk_dim) in NLC
        speaker_emb = self.speaker_encoder(mel_ref)
        # Transpose to NCL: (batch, spk_dim, 1) for broadcasting with x in NCL
        speaker_emb = speaker_emb.transpose(0, 2, 1)  # (batch, spk_dim, 1)

        # x input is NLC (batch, seq_len, gpt_dim), convert to NCL
        x = x.transpose(0, 2, 1)  # (batch, gpt_dim, seq_len) NCL format

        # Pre-conv: NCL -> NCL
        # MLX Conv1d expects NLC, so transpose
        x = x.transpose(0, 2, 1)  # NCL -> NLC
        x = self.conv_pre(x)
        x = x.transpose(0, 2, 1)  # NLC -> NCL

        # Add speaker conditioning: cond_layer expects NLC
        cond = speaker_emb.transpose(0, 2, 1)  # NCL -> NLC
        cond = self.cond_layer(cond)
        cond = cond.transpose(0, 2, 1)  # NLC -> NCL
        x = x + cond

        # Upsampling with residual blocks
        for i in range(self.num_upsamples):
            # Upsample (ConvTranspose1d)
            for up in self.ups[i]:
                x = x.transpose(0, 2, 1)  # NCL -> NLC
                x = up(x)
                x = x.transpose(0, 2, 1)  # NLC -> NCL

            # Add per-layer conditioning
            if self.conds:
                cond = speaker_emb.transpose(0, 2, 1)  # NCL -> NLC
                cond = self.conds[i](cond)
                cond = cond.transpose(0, 2, 1)  # NLC -> NCL
                x = x + cond

            # Apply residual blocks (they expect NCL format now)
            xs = None
            for j in range(self.num_kernels):
                block_idx = i * self.num_kernels + j
                # AMPBlock expects NCL format directly
                res = self.resblocks[block_idx](x)
                if xs is None:
                    xs = res
                else:
                    xs = xs + res

            x = xs / self.num_kernels

        # Post-processing
        # Snake activation expects NCL format (already in NCL)
        x = self.activation_post(x)

        # Post conv
        x = x.transpose(0, 2, 1)  # NCL -> NLC for Conv1d
        x = self.conv_post(x)
        x = x.transpose(0, 2, 1)  # NLC -> NCL

        # Always use tanh (matching PyTorch implementation)
        x = mx.tanh(x)

        return x
