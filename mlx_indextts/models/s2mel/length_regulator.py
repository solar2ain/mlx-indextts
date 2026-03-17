"""Length Regulator (InterpolateRegulator) for S2Mel.

Handles upsampling/interpolation of semantic codes to mel timesteps.
"""

from typing import Tuple, Optional

import mlx.core as mx
import mlx.nn as nn


class InterpolateRegulator(nn.Module):
    """Length regulator using interpolation for upsampling.

    This module takes semantic codes and upsamples them to match the target
    mel spectrogram length using nearest-neighbor interpolation and
    conv-norm-activation layers.

    The internal processing uses NCL format to match PyTorch behavior.

    Args:
        channels: Hidden dimension size (512)
        sampling_ratios: List of sampling ratios (defines number of conv-norm-act blocks)
        is_discrete: Whether input is discrete codes (False for IndexTTS 2.0)
        in_channels: Input dimension for continuous input (1024)
        codebook_size: Size of codebook for discrete input (2048)
        n_codebooks: Number of codebooks (not used)
        out_channels: Output dimension (defaults to channels)
    """

    def __init__(
        self,
        channels: int = 512,
        sampling_ratios: Tuple[int, ...] = (1, 1, 1, 1),
        is_discrete: bool = False,
        in_channels: Optional[int] = None,
        codebook_size: int = 1024,
        out_channels: Optional[int] = None,
        groups: int = 1,
        n_codebooks: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        self.channels = channels
        out_channels = out_channels or channels
        self.is_discrete = is_discrete
        self.groups = groups if groups > 0 else 1

        # Build conv-norm-act layers as a Sequential-like list
        # PyTorch structure: [Conv, GroupNorm, Mish] * len(sampling_ratios) + [Conv1x1]
        self.model = []
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                # Conv1d 3x1 (stored with MLX format: out, kernel, in)
                self.model.append(nn.Conv1d(channels, channels, kernel_size=3, padding=1))
                self.model.append(GroupNorm(groups, channels))
                self.model.append(Mish())
        else:
            self.interpolate = False

        # Final 1x1 projection
        self.model.append(nn.Conv1d(channels, out_channels, kernel_size=1))

        # Embedding for discrete input
        self.embedding = nn.Embedding(codebook_size, channels)

        # Input projection for continuous input
        if not is_discrete:
            self.content_in_proj = nn.Linear(in_channels, channels)

        # Mask token for padding
        self.mask_token = mx.zeros((1, channels))

    def __call__(
        self,
        x: mx.array,
        ylens: mx.array,
        n_quantizers: Optional[int] = None,
        f0: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, None, None, None]:
        """Forward pass.

        Args:
            x: Input semantic codes/features (batch, seq_len, dim) for continuous,
               or (batch, seq_len) for discrete
            ylens: Target lengths (batch,)
            n_quantizers: Number of quantizers to use (not used)
            f0: F0 conditioning (not used in IndexTTS 2.0)

        Returns:
            Tuple of (output, output_lengths, None, None, None)
            output: Upsampled features (batch, target_len, out_channels)
        """
        # Handle discrete vs continuous input
        if self.is_discrete:
            x = self.embedding(x)
        else:
            x = self.content_in_proj(x)

        # x is now (batch, seq_len, channels) - NLC format

        # Get target length
        max_len = int(ylens.max().item())

        if self.interpolate:
            # Transpose to NCL for interpolation (matches PyTorch)
            x = x.transpose(0, 2, 1)  # NLC -> NCL
            x = _interpolate_nearest(x, max_len)  # (batch, channels, max_len)
        else:
            x = x.transpose(0, 2, 1)  # NLC -> NCL

        # Apply conv-norm-act layers (NCL format internally)
        for layer in self.model:
            if isinstance(layer, nn.Conv1d):
                # MLX Conv1d expects NLC, so transpose around it
                x = x.transpose(0, 2, 1)  # NCL -> NLC
                x = layer(x)
                x = x.transpose(0, 2, 1)  # NLC -> NCL
            elif isinstance(layer, GroupNorm):
                x = layer(x)  # NCL format
            else:
                x = layer(x)  # Mish works on any format

        # Transpose back to NLC format for output
        out = x.transpose(0, 2, 1)  # NCL -> NLC

        # Apply sequence mask
        mask = _sequence_mask(ylens, max_len)[:, :, None]  # (batch, max_len, 1)
        out = out * mask

        return out, ylens, None, None, None


class GroupNorm(nn.Module):
    """Group Normalization layer.

    Expects input in NCL format (batch, channels, length).
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups if num_groups > 0 else 1
        self.num_channels = num_channels
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor in NCL format (batch, channels, length)

        Returns:
            Normalized tensor in NCL format
        """
        batch, channels, length = x.shape

        # Reshape for group norm: (batch, groups, channels_per_group, length)
        x = x.reshape(batch, self.num_groups, channels // self.num_groups, length)

        # Compute mean and variance over (channels_per_group, length)
        mean = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True)

        # Normalize
        x = (x - mean) / mx.sqrt(var + self.eps)

        # Reshape back
        x = x.reshape(batch, channels, length)

        # Apply affine transformation
        x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.tanh(mx.log(1 + mx.exp(x)))


def _sequence_mask(lengths: mx.array, max_length: Optional[int] = None) -> mx.array:
    """Create a boolean mask from sequence lengths.

    Args:
        lengths: Tensor of lengths (batch,)
        max_length: Maximum length (defaults to max of lengths)

    Returns:
        Boolean mask (batch, max_length)
    """
    if max_length is None:
        max_length = int(lengths.max().item())

    # Create range tensor
    seq_range = mx.arange(max_length)[None, :]  # (1, max_length)

    # Compare with lengths
    mask = seq_range < lengths[:, None]  # (batch, max_length)

    return mask.astype(mx.float32)


def _interpolate_nearest(x: mx.array, target_length: int) -> mx.array:
    """Nearest neighbor interpolation.

    Args:
        x: Input tensor in NCL format (batch, channels, length)
        target_length: Target output length

    Returns:
        Interpolated tensor (batch, channels, target_length)
    """
    batch, channels, length = x.shape

    # Compute source indices for nearest neighbor
    scale = length / target_length
    indices = (mx.arange(target_length) * scale).astype(mx.int32)
    indices = mx.clip(indices, 0, length - 1)

    # Gather
    out = x[:, :, indices]

    return out
