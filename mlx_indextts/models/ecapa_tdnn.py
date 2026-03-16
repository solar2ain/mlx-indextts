"""ECAPA-TDNN speaker encoder for BigVGAN conditioning.

Structure matches original PyTorch implementation:
- blocks: [TDNNBlock, SERes2NetBlock, SERes2NetBlock, SERes2NetBlock]
- mfa: TDNNBlock
- asp: AttentiveStatisticsPooling
- asp_bn: BatchNorm1d
- fc: Conv1d
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def reflect_pad_1d(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """Apply reflect padding to 1D signal.

    Args:
        x: Input (batch, time, channels) - NLC format
        pad_left: Left padding size
        pad_right: Right padding size

    Returns:
        Padded tensor (batch, time + pad_left + pad_right, channels)
    """
    # Use numpy for reflect padding (MLX doesn't support it natively)
    x_np = np.array(x)
    x_padded = np.pad(x_np, ((0, 0), (pad_left, pad_right), (0, 0)), mode='reflect')
    return mx.array(x_padded)


class TDNNBlock(nn.Module):
    """Time-Delay Neural Network block.

    Structure: conv -> activation -> norm
    Parameter names: conv.weight, conv.bias, norm.weight, norm.bias, norm.running_mean, norm.running_var

    Uses reflect padding to match PyTorch SpeechBrain implementation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Calculate padding for 'same' output
        self.pad_size = (kernel_size - 1) // 2 * dilation

        # Conv1d with no padding (we'll handle it manually with reflect)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # No padding, we do reflect padding manually
        )
        self.norm = nn.BatchNorm(out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - NCL format (PyTorch convention)

        Returns:
            Output (batch, out_channels, time)

        Note: Order is conv -> activation -> norm (matching PyTorch)
        """
        # Convert NCL -> NLC for padding and conv
        x = x.transpose(0, 2, 1)

        # Apply reflect padding (matching PyTorch SpeechBrain)
        if self.pad_size > 0:
            x = reflect_pad_1d(x, self.pad_size, self.pad_size)

        x = self.conv(x)
        x = nn.relu(x)  # Activation BEFORE norm (PyTorch order)
        x = self.norm(x)

        # Convert back NLC -> NCL
        x = x.transpose(0, 2, 1)
        return x


class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale feature extraction.

    Uses a list of TDNNBlocks named 'blocks'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        self.scale = scale
        in_channel = in_channels // scale
        hidden_channel = out_channels // scale

        # Named 'blocks' to match original structure
        self.blocks = [
            TDNNBlock(
                in_channel,
                hidden_channel,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            for _ in range(scale - 1)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - NCL format

        Returns:
            Output (batch, channels, time)
        """
        # Split along channel dimension
        width = x.shape[1] // self.scale
        y = []

        for i in range(self.scale):
            x_i = x[:, i * width:(i + 1) * width, :]

            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)

            y.append(y_i)

        return mx.concatenate(y, axis=1)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Structure: conv1 -> relu -> conv2 -> sigmoid
    """

    def __init__(self, in_channels: int, se_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, se_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(se_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array, lengths: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - NCL format
            lengths: Optional sequence lengths (not used in inference)

        Returns:
            Scaled output (batch, channels, time)
        """
        # Global mean pooling over time
        # Convert NCL -> NLC for MLX conv
        x_nlc = x.transpose(0, 2, 1)
        s = mx.mean(x_nlc, axis=1, keepdims=True)  # (batch, 1, channels)

        s = self.conv1(s)
        s = nn.relu(s)
        s = self.conv2(s)
        s = mx.sigmoid(s)

        # s is (batch, 1, channels), convert to NCL: (batch, channels, 1)
        s = s.transpose(0, 2, 1)

        return s * x


class SERes2NetBlock(nn.Module):
    """SE-Res2Net block used in ECAPA-TDNN.

    Structure: tdnn1 -> res2net_block -> tdnn2 -> se_block + skip
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels

        # First TDNN (1x1 conv)
        self.tdnn1 = TDNNBlock(
            in_channels, out_channels, kernel_size=1, dilation=1
        )

        # Res2Net block
        self.res2net_block = Res2NetBlock(
            out_channels, out_channels, res2net_scale, kernel_size, dilation
        )

        # Second TDNN (1x1 conv)
        self.tdnn2 = TDNNBlock(
            out_channels, out_channels, kernel_size=1, dilation=1
        )

        # SE block
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        # Skip connection if dimensions differ
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def __call__(self, x: mx.array, lengths: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - NCL format
            lengths: Optional sequence lengths

        Returns:
            Output (batch, out_channels, time)
        """
        residual = x
        if self.shortcut is not None:
            # Convert for MLX conv: NCL -> NLC -> conv -> NLC -> NCL
            residual = residual.transpose(0, 2, 1)
            residual = self.shortcut(residual)
            residual = residual.transpose(0, 2, 1)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, lengths)

        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling layer.

    Structure: tdnn (with global context) -> tanh -> conv -> softmax -> weighted stats
    """

    def __init__(
        self,
        channels: int,
        attention_channels: int = 128,
        global_context: bool = True,
    ):
        super().__init__()
        self.global_context = global_context
        self.eps = 1e-12

        # Input channels: channels * 3 if global_context else channels
        input_ch = channels * 3 if global_context else channels
        self.tdnn = TDNNBlock(input_ch, attention_channels, kernel_size=1, dilation=1)
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def __call__(self, x: mx.array, lengths: Optional[mx.array] = None) -> mx.array:
        """Forward pass.

        Args:
            x: Input (batch, channels, time) - NCL format
            lengths: Optional sequence lengths

        Returns:
            Pooled output (batch, channels * 2, 1) - mean and std concatenated
        """
        L = x.shape[-1]
        batch_size = x.shape[0]

        # Create mask
        if lengths is None:
            lengths = mx.ones((batch_size,))

        # Mask: (batch, 1, time)
        mask = mx.arange(L)[None, None, :] < (lengths[:, None, None] * L)
        mask = mask.astype(x.dtype)

        total = mx.sum(mask, axis=2, keepdims=True)  # (batch, 1, 1)

        if self.global_context:
            # Compute mean and std with mask
            mean = mx.sum(mask * x, axis=2, keepdims=True) / (total + self.eps)
            std = mx.sqrt(
                mx.sum(mask * (x - mean) ** 2, axis=2, keepdims=True) / (total + self.eps) + self.eps
            )

            # Expand to time dimension
            mean = mx.broadcast_to(mean, x.shape)
            std = mx.broadcast_to(std, x.shape)

            # Concatenate: (batch, channels*3, time)
            attn = mx.concatenate([x, mean, std], axis=1)
        else:
            attn = x

        # Apply TDNN, tanh, and conv (PyTorch: self.conv(self.tanh(self.tdnn(attn))))
        attn = self.tdnn(attn)  # (batch, attention_channels, time)
        attn = mx.tanh(attn)  # tanh activation before conv

        # Convert for MLX conv: NCL -> NLC
        attn = attn.transpose(0, 2, 1)
        attn = self.conv(attn)  # (batch, time, channels)
        attn = attn.transpose(0, 2, 1)  # Back to NCL: (batch, channels, time)

        # Mask and softmax
        attn = mx.where(mask > 0, attn, mx.array(float("-inf")))
        attn = mx.softmax(attn, axis=2)  # (batch, channels, time)

        # Weighted statistics
        mean = mx.sum(attn * x, axis=2, keepdims=True)  # (batch, channels, 1)
        # Use numerically stable variance computation
        # var = E[x^2] - E[x]^2, but clip to avoid negative values due to numerical precision
        variance = mx.sum(attn * x ** 2, axis=2, keepdims=True) - mean ** 2
        variance = mx.maximum(variance, 0.0)  # Clip negative values
        std = mx.sqrt(variance + self.eps)  # (batch, channels, 1)

        # Concatenate mean and std: (batch, channels*2, 1)
        pooled_stats = mx.concatenate([mean, std], axis=1)

        return pooled_stats


class ECAPATDNN(nn.Module):
    """ECAPA-TDNN speaker encoder.

    Structure matches original:
    - blocks: [TDNNBlock, SERes2NetBlock x 3]
    - mfa: TDNNBlock
    - asp: AttentiveStatisticsPooling
    - asp_bn: BatchNorm
    - fc: Conv1d
    """

    def __init__(
        self,
        input_size: int = 100,
        lin_neurons: int = 192,
        channels: list = [512, 512, 512, 512, 1536],
        kernel_sizes: list = [5, 3, 3, 3, 1],
        dilations: list = [1, 2, 3, 4, 1],
        attention_channels: int = 128,
        res2net_scale: int = 8,
        se_channels: int = 128,
        global_context: bool = True,
    ):
        """Initialize ECAPA-TDNN.

        Args:
            input_size: Number of mel bins
            lin_neurons: Output embedding dimension
            channels: Channel sizes for each block
            kernel_sizes: Kernel sizes for each block
            dilations: Dilation rates for each block
            attention_channels: Attention channels in ASP
            res2net_scale: Scale factor for Res2Net
            se_channels: SE block reduction channels
            global_context: Use global context in ASP
        """
        super().__init__()
        self.input_size = input_size
        self.channels = channels

        # Build blocks
        self.blocks = []

        # First block: initial TDNN
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
            )
        )

        # SE-Res2Net layers (blocks 1-3)
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                )
            )

        # Multi-layer feature aggregation
        # Input: concatenation of outputs from blocks 1-3
        mfa_input_ch = channels[-2] * (len(channels) - 2)  # 512 * 3 = 1536
        self.mfa = TDNNBlock(
            mfa_input_ch,
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
        )

        # Attentive Statistics Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )

        # Batch norm after ASP
        self.asp_bn = nn.BatchNorm(channels[-1] * 2)

        # Final projection (Conv1d with kernel_size=1)
        self.fc = nn.Conv1d(channels[-1] * 2, lin_neurons, kernel_size=1)

    def __call__(
        self,
        x: mx.array,
        lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """Extract speaker embedding.

        Args:
            x: Input mel spectrogram - can be either:
               - (batch, time, n_mels) - NLC format
               - (batch, n_mels, time) - NCL format (PyTorch convention)
            lengths: Optional sequence lengths (relative, 0-1)

        Returns:
            Speaker embedding (batch, 1, lin_neurons)
        """
        # Ensure correct shape
        if x.ndim == 2:
            x = x[None, :, :]

        # Detect input format and convert to NCL (batch, n_mels, time)
        # Original PyTorch expects NLC and does: x = x.transpose(1, 2)
        # If input shape is (batch, n_mels, time) where n_mels == input_size,
        # it's already in NCL format
        if x.shape[1] == self.input_size:
            # Already NCL format: (batch, n_mels, time)
            pass
        elif x.shape[-1] == self.input_size:
            # NLC format: (batch, time, n_mels) -> transpose to NCL
            x = x.transpose(0, 2, 1)  # (batch, n_mels, time)
        else:
            raise ValueError(
                f"Input shape {x.shape} doesn't match input_size={self.input_size}. "
                f"Expected either (batch, {self.input_size}, time) or (batch, time, {self.input_size})"
            )

        # Process through blocks
        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        # Concatenate outputs from blocks 1-3 (skip initial TDNN output)
        x = mx.concatenate(xl[1:], axis=1)  # (batch, 1536, time)
        x = self.mfa(x)

        # Attentive Statistics Pooling
        x = self.asp(x, lengths)  # (batch, 3072, 1)

        # Batch norm - convert to NLC for BatchNorm
        x = x.transpose(0, 2, 1)  # (batch, 1, 3072)
        x = self.asp_bn(x)

        # Final projection
        x = self.fc(x)  # (batch, 1, lin_neurons)

        return x
