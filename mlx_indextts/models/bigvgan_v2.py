"""BigVGAN v2 vocoder for IndexTTS 2.0.

This is a pure vocoder (no speaker encoder) that converts mel spectrogram to audio.
Uses nvidia/bigvgan_v2_22khz_80band_256x pretrained weights.

Key differences from BigVGAN 1.5:
- Input: mel spectrogram (num_mels=80) instead of GPT latent (gpt_dim)
- No speaker encoder (ECAPA-TDNN)
- No conditioning injection
- use_tanh_at_final=False, use_bias_at_final=False
"""

from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_indextts.models.activations import (
    Activation1d,
    Snake,
    SnakeBeta,
)


@dataclass
class BigVGANV2Config:
    """Configuration for BigVGAN v2.

    Default values match nvidia/bigvgan_v2_22khz_80band_256x.
    """
    num_mels: int = 80
    upsample_rates: List[int] = None
    upsample_kernel_sizes: List[int] = None
    upsample_initial_channel: int = 1536
    resblock_kernel_sizes: List[int] = None
    resblock_dilation_sizes: List[List[int]] = None
    activation: str = "snakebeta"
    snake_logscale: bool = True
    use_tanh_at_final: bool = False
    use_bias_at_final: bool = False
    resblock: str = "1"  # "1" for AMPBlock1, "2" for AMPBlock2

    def __post_init__(self):
        if self.upsample_rates is None:
            self.upsample_rates = [4, 4, 2, 2, 2, 2]
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [8, 8, 4, 4, 4, 4]
        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11]
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    @classmethod
    def from_dict(cls, d: dict) -> "BigVGANV2Config":
        return cls(
            num_mels=d.get("num_mels", 80),
            upsample_rates=d.get("upsample_rates"),
            upsample_kernel_sizes=d.get("upsample_kernel_sizes"),
            upsample_initial_channel=d.get("upsample_initial_channel", 1536),
            resblock_kernel_sizes=d.get("resblock_kernel_sizes"),
            resblock_dilation_sizes=d.get("resblock_dilation_sizes"),
            activation=d.get("activation", "snakebeta"),
            snake_logscale=d.get("snake_logscale", True),
            use_tanh_at_final=d.get("use_tanh_at_final", False),
            use_bias_at_final=d.get("use_bias_at_final", False),
            resblock=d.get("resblock", "1"),
        )


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for same output length."""
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    """Anti-aliased Multi-Periodicity block (Type 1).

    Contains two sets of convolutions: convs1 with various dilations,
    and convs2 with fixed dilation=1 after each convs1 layer.

    Input/Output: NCL format (batch, channels, length).
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

        # convs1: dilated convolutions
        self.convs1 = []
        for d in dilations:
            padding = get_padding(kernel_size, d)
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding)
            )

        # convs2: fixed dilation=1 after each convs1
        self.convs2 = []
        for _ in dilations:
            padding = get_padding(kernel_size, 1)
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=padding)
            )

        # Activations: 2 per dilation (before convs1, before convs2)
        num_layers = len(self.convs1) + len(self.convs2)
        if activation == "snakebeta":
            self.activations = [
                Activation1d(SnakeBeta(channels, alpha_logscale))
                for _ in range(num_layers)
            ]
        elif activation == "snake":
            self.activations = [
                Activation1d(Snake(channels, alpha_logscale))
                for _ in range(num_layers)
            ]
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, length) NCL format

        Returns:
            Output (batch, channels, length) NCL format
        """
        acts1 = self.activations[::2]   # even indices: 0, 2, 4
        acts2 = self.activations[1::2]  # odd indices: 1, 3, 5

        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            # activation -> conv1 -> activation -> conv2 -> residual
            xt = a1(x)  # NCL (activation expects NCL)

            # Conv1d in MLX uses NLC format
            xt = xt.transpose(0, 2, 1)  # NCL -> NLC
            xt = c1(xt)
            xt = xt.transpose(0, 2, 1)  # NLC -> NCL

            xt = a2(xt)  # NCL

            xt = xt.transpose(0, 2, 1)  # NCL -> NLC
            xt = c2(xt)
            xt = xt.transpose(0, 2, 1)  # NLC -> NCL

            x = xt + x

        return x


class AMPBlock2(nn.Module):
    """Anti-aliased Multi-Periodicity block (Type 2).

    Simpler version without the extra convs2 layer.
    Only convs with various dilations.

    Input/Output: NCL format (batch, channels, length).
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

        # Single set of dilated convolutions
        self.convs = []
        for d in dilations:
            padding = get_padding(kernel_size, d)
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=padding)
            )

        # One activation per conv
        num_layers = len(self.convs)
        if activation == "snakebeta":
            self.activations = [
                Activation1d(SnakeBeta(channels, alpha_logscale))
                for _ in range(num_layers)
            ]
        elif activation == "snake":
            self.activations = [
                Activation1d(Snake(channels, alpha_logscale))
                for _ in range(num_layers)
            ]
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, length) NCL format

        Returns:
            Output (batch, channels, length) NCL format
        """
        for conv, act in zip(self.convs, self.activations):
            xt = act(x)  # NCL

            xt = xt.transpose(0, 2, 1)  # NCL -> NLC
            xt = conv(xt)
            xt = xt.transpose(0, 2, 1)  # NLC -> NCL

            x = xt + x

        return x


class BigVGANV2(nn.Module):
    """BigVGAN v2 vocoder.

    Pure mel-to-audio vocoder without speaker conditioning.
    Uses anti-aliased periodic activations for high-quality audio.

    Input: mel spectrogram (batch, n_mels, time) NCL format
    Output: audio waveform (batch, 1, samples) NCL format
    """

    def __init__(self, config: BigVGANV2Config):
        """Initialize BigVGAN v2.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # Select resblock type
        if config.resblock == "1":
            resblock_class = AMPBlock1
        elif config.resblock == "2":
            resblock_class = AMPBlock2
        else:
            raise ValueError(f"Unknown resblock type: {config.resblock}")

        # Pre-convolution: (n_mels) -> (upsample_initial_channel)
        # kernel=7, padding=3 for same output length
        self.conv_pre = nn.Conv1d(
            config.num_mels,
            config.upsample_initial_channel,
            kernel_size=7,
            padding=3,
        )

        # Upsampling layers with transposed convolutions
        self.ups = []
        ch = config.upsample_initial_channel
        for i, (rate, kernel) in enumerate(zip(
            config.upsample_rates,
            config.upsample_kernel_sizes
        )):
            out_ch = ch // 2
            padding = (kernel - rate) // 2
            self.ups.append(
                nn.ConvTranspose1d(ch, out_ch, kernel, stride=rate, padding=padding)
            )
            ch = out_ch

        # Residual blocks
        self.resblocks = []
        ch = config.upsample_initial_channel
        for i in range(self.num_upsamples):
            ch = ch // 2
            for k, d in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes
            ):
                self.resblocks.append(
                    resblock_class(
                        ch,
                        kernel_size=k,
                        dilations=d,
                        activation=config.activation,
                        alpha_logscale=config.snake_logscale,
                    )
                )

        # Post activation (wrapped with Activation1d for anti-aliasing)
        if config.activation == "snakebeta":
            self.activation_post = Activation1d(
                SnakeBeta(ch, config.snake_logscale)
            )
        else:
            self.activation_post = Activation1d(
                Snake(ch, config.snake_logscale)
            )

        # Post convolution: (ch) -> (1)
        # kernel=7, padding=3, bias depends on config
        self.conv_post = nn.Conv1d(
            ch, 1, kernel_size=7, padding=3, bias=config.use_bias_at_final
        )

        self.use_tanh = config.use_tanh_at_final

    def __call__(self, x: mx.array) -> mx.array:
        """Generate audio from mel spectrogram.

        Args:
            x: Mel spectrogram (batch, n_mels, time) NCL format

        Returns:
            Audio waveform (batch, 1, samples) NCL format
        """
        # Pre-conv: NCL input
        # MLX Conv1d expects NLC, so transpose
        x = x.transpose(0, 2, 1)  # NCL -> NLC
        x = self.conv_pre(x)
        x = x.transpose(0, 2, 1)  # NLC -> NCL

        # Upsample + resblocks
        for i in range(self.num_upsamples):
            # Upsample
            x = x.transpose(0, 2, 1)  # NCL -> NLC
            x = self.ups[i](x)
            x = x.transpose(0, 2, 1)  # NLC -> NCL

            # Apply residual blocks and average
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                res = self.resblocks[idx](x)
                if xs is None:
                    xs = res
                else:
                    xs = xs + res
            x = xs / self.num_kernels

        # Post processing
        x = self.activation_post(x)  # NCL

        x = x.transpose(0, 2, 1)  # NCL -> NLC
        x = self.conv_post(x)
        x = x.transpose(0, 2, 1)  # NLC -> NCL

        # Final activation
        if self.use_tanh:
            x = mx.tanh(x)
        else:
            # Clamp to [-1, 1]
            x = mx.clip(x, -1.0, 1.0)

        return x
