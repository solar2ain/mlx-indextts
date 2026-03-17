"""WaveNet module for S2Mel final layer.

This implements the WaveNet-style final layer used in the DiT model.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class WN(nn.Module):
    """WaveNet module with dilated convolutions and gated activations.

    Args:
        hidden_channels: Hidden dimension (512)
        kernel_size: Convolution kernel size (5)
        dilation_rate: Base dilation rate (1)
        n_layers: Number of layers (8)
        gin_channels: Conditioning channels (512)
        p_dropout: Dropout probability (0.2)
        causal: Whether to use causal convolutions (False)
    """

    def __init__(
        self,
        hidden_channels: int = 512,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        n_layers: int = 8,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.causal = causal

        self.in_layers = []
        self.res_skip_layers = []

        if gin_channels != 0:
            self.cond_layer = SConv1d(
                gin_channels,
                2 * hidden_channels * n_layers,
                kernel_size=1,
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)

            in_layer = SConv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.in_layers.append(in_layer)

            # Last layer only outputs hidden_channels (no res)
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = SConv1d(
                hidden_channels,
                res_skip_channels,
                kernel_size=1,
            )
            self.res_skip_layers.append(res_skip_layer)

        self.drop = nn.Dropout(p_dropout) if p_dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        x_mask: mx.array,
        g: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, channels, length) - NCL format
            x_mask: Mask tensor (batch, 1, length)
            g: Optional conditioning (batch, gin_channels, 1) or (batch, gin_channels)

        Returns:
            Output tensor (batch, channels, length) - NCL format
        """
        output = mx.zeros_like(x)

        if g is not None:
            # Ensure g has 3 dimensions
            if g.ndim == 2:
                g = g[:, :, None]  # (batch, gin_channels) -> (batch, gin_channels, 1)
            g = self.cond_layer(g)  # (batch, 2*hidden*n_layers, 1)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)  # (batch, 2*hidden, length)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = mx.zeros_like(x_in)

            # Gated activation: tanh(a) * sigmoid(b)
            acts = _fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)

            if self.drop is not None:
                acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                # Split into residual and skip
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts

        return output * x_mask


class SConv1d(nn.Module):
    """Conv1d with weight normalization and symmetric padding.

    This is a simplified version of the encodec SConv1d that handles
    the weight-normalized convolutions used in WaveNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Calculate padding if not provided
        if padding is None:
            # Default: same padding
            effective_kernel = (kernel_size - 1) * dilation + 1
            padding = effective_kernel // 2

        self.padding = padding

        # Conv1d weight: (out_channels, kernel_size, in_channels) in MLX NKI format
        # We'll store in OIK format and transpose during forward
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # We do manual padding
        )

        # Store dilation for manual application
        self._dilation = dilation

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, channels, length) - NCL format

        Returns:
            Output tensor (batch, out_channels, length) - NCL format
        """
        # Handle dilation by inserting zeros in kernel (or use dilated conv directly)
        # For simplicity, we'll do manual padding and use the conv with dilation

        # Convert NCL -> NLC for MLX conv
        x = x.transpose(0, 2, 1)

        # Calculate padding for dilated conv
        effective_kernel = (self.kernel_size - 1) * self._dilation + 1
        total_padding = effective_kernel - 1  # For 'same' output

        # Symmetric padding
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left

        # Manual padding
        if pad_left > 0 or pad_right > 0:
            x = _pad1d(x, pad_left, pad_right)

        # Apply convolution
        # MLX doesn't directly support dilation in nn.Conv1d, so we need a workaround
        # For now, if dilation > 1, we manually dilate the kernel
        if self._dilation > 1:
            # Get weight and manually apply dilated convolution
            weight = self.conv.weight  # (out_channels, kernel_size, in_channels)
            bias = self.conv.bias if hasattr(self.conv, 'bias') else None

            # Create dilated kernel using functional approach
            out = _dilated_conv1d(x, weight, bias, self.stride, self._dilation)
        else:
            out = self.conv(x)

        # Convert NLC -> NCL
        out = out.transpose(0, 2, 1)

        return out


def _fused_add_tanh_sigmoid_multiply(
    input_a: mx.array,
    input_b: mx.array,
    n_channels: int,
) -> mx.array:
    """Fused add + tanh + sigmoid + multiply operation.

    Args:
        input_a: First input (batch, 2*n_channels, length)
        input_b: Second input (batch, 2*n_channels, length)
        n_channels: Number of channels per half

    Returns:
        Output (batch, n_channels, length)
    """
    in_act = input_a + input_b

    # Split into two halves
    t_act = mx.tanh(in_act[:, :n_channels, :])
    s_act = mx.sigmoid(in_act[:, n_channels:, :])

    return t_act * s_act


def _pad1d(x: mx.array, pad_left: int, pad_right: int, mode: str = 'reflect') -> mx.array:
    """Pad 1D tensor.

    Args:
        x: Input tensor (batch, length, channels) - NLC format
        pad_left: Left padding
        pad_right: Right padding
        mode: Padding mode ('reflect' or 'constant')

    Returns:
        Padded tensor
    """
    if pad_left == 0 and pad_right == 0:
        return x

    # Use numpy for reflect padding (MLX doesn't support it directly)
    if mode == 'reflect':
        x_np = np.array(x)
        x_padded = np.pad(x_np, ((0, 0), (pad_left, pad_right), (0, 0)), mode='reflect')
        return mx.array(x_padded)
    else:
        # Zero padding
        batch, length, channels = x.shape
        zeros_left = mx.zeros((batch, pad_left, channels))
        zeros_right = mx.zeros((batch, pad_right, channels))
        return mx.concatenate([zeros_left, x, zeros_right], axis=1)


def _dilated_conv1d(
    x: mx.array,
    weight: mx.array,
    bias: Optional[mx.array],
    stride: int,
    dilation: int,
) -> mx.array:
    """Apply dilated convolution manually.

    Args:
        x: Input (batch, length, in_channels) - NLC format
        weight: Conv weight (out_channels, kernel_size, in_channels)
        bias: Optional bias (out_channels,)
        stride: Stride
        dilation: Dilation factor

    Returns:
        Output (batch, out_length, out_channels) - NLC format
    """
    batch, length, in_channels = x.shape
    out_channels, kernel_size, _ = weight.shape

    # Dilate the kernel by inserting zeros
    if dilation > 1:
        # Create dilated kernel
        dilated_size = (kernel_size - 1) * dilation + 1
        dilated_weight = mx.zeros((out_channels, dilated_size, in_channels))

        # Copy weights at dilated positions
        weight_np = np.array(weight)
        dilated_weight_np = np.zeros((out_channels, dilated_size, in_channels))
        for k in range(kernel_size):
            dilated_weight_np[:, k * dilation, :] = weight_np[:, k, :]
        dilated_weight = mx.array(dilated_weight_np)
    else:
        dilated_weight = weight

    # Apply convolution using mx.conv1d
    # mx.conv1d expects (N, L, C_in) input and (C_out, K, C_in) weight
    out = mx.conv1d(x, dilated_weight, stride=stride)

    if bias is not None:
        out = out + bias[None, None, :]

    return out


class LayerNorm(nn.Module):
    """Layer normalization for convolutional layers.

    Expects input in NCL format (batch, channels, length).
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = mx.ones((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor in NCL format (batch, channels, length)

        Returns:
            Normalized tensor in NCL format
        """
        # NCL -> NLC for layer norm
        x = x.transpose(0, 2, 1)

        # Apply layer norm over channels dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = x * self.gamma[None, None, :] + self.beta[None, None, :]

        # NLC -> NCL
        x = x.transpose(0, 2, 1)

        return x
