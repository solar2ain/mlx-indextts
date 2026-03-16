"""Conformer encoder for IndexTTS conditioning."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.config import ConformerConfig


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed forward layer."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_1 = nn.Linear(dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.activation = nn.SiLU()

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        x = self.w_1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.w_2(x)
        return x


class ConvolutionModule(nn.Module):
    """Convolution module in Conformer."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = channels

        # MLX Conv1d expects NLC format (batch, length, channels)
        # Pointwise conv (expand to 2x)
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1)

        # Depthwise conv
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )

        # LayerNorm
        self.norm = nn.LayerNorm(channels)

        # Pointwise conv (back to original)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1)

        self.activation = nn.SiLU()

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, channels) - NLC format
            mask: Padding mask (batch, 1, seq_len)

        Returns:
            Output tensor (batch, seq_len, channels)
        """
        # x is already in NLC format for MLX

        # GLU: pointwise conv then split and gate
        x = self.pointwise_conv1(x)  # (batch, seq_len, 2*channels)
        # GLU split
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * mx.sigmoid(x2)

        # Depthwise conv
        x = self.depthwise_conv(x)

        # LayerNorm and activation (already in NLC format)
        x = self.norm(x)
        x = self.activation(x)

        # Final pointwise conv
        x = self.pointwise_conv2(x)

        return x


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding.

    Note: This implements the xscale factor (sqrt(d_model)) that scales
    the input before adding positional encoding. This is critical for
    matching PyTorch's implementation.
    """

    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        self.xscale = math.sqrt(dim)  # Critical: scale factor
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Pre-compute PE
        self._pe = None
        self.max_len = max_len

    def _compute_pe(self, max_len: int) -> mx.array:
        """Compute sinusoidal positional encoding."""
        import numpy as np

        pe = np.zeros((max_len, self.dim), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.dim, 2, dtype=np.float32) * -(np.log(10000.0) / self.dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return mx.array(pe)

    @property
    def pe(self) -> mx.array:
        if self._pe is None:
            self._pe = self._compute_pe(self.max_len)
        return self._pe

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Apply positional encoding with xscale.

        Args:
            x: Input (batch, seq_len, dim)

        Returns:
            Tuple of (x * xscale, pe) - Note: x is scaled but pe is NOT added
        """
        seq_len = x.shape[1]
        pe = self.pe[:seq_len][None, :, :]

        # Scale input by sqrt(dim) - critical for matching PyTorch!
        x = x * self.xscale

        if self.dropout is not None:
            x = self.dropout(x)
            pe = self.dropout(pe)

        return x, pe


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head attention with relative position encoding.

    Matches PyTorch wenet/conformer implementation.
    Note: rel_shift is NOT used (commented out in original PyTorch code).
    """

    def __init__(self, num_heads: int, dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)
        self.linear_pos = nn.Linear(dim, dim, bias=False)

        # Position bias
        self.pos_bias_u = mx.zeros((num_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array,
        pos_emb: mx.array,
        cache: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Matches PyTorch: RelPositionMultiHeadedAttention.forward()
        """
        b, seq_len, _ = query.shape

        # Linear projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # Reshape to (batch, seq, heads, head_dim)
        q = q.reshape(b, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(b, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(b, seq_len, self.num_heads, self.head_dim)

        # k, v: transpose to (batch, heads, seq, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Position encoding: (batch, seq, dim) -> (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb)
        p = p.reshape(n_batch_pos, -1, self.num_heads, self.head_dim)
        p = p.transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        # Add position bias to q BEFORE transpose (PyTorch order)
        # q is (batch, seq, heads, head_dim), pos_bias is (heads, head_dim)
        # Broadcasting: (batch, seq, heads, head_dim) + (heads, head_dim) -> need to align dims
        q_with_bias_u = (q + self.pos_bias_u[None, None, :, :]).transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)
        q_with_bias_v = (q + self.pos_bias_v[None, None, :, :]).transpose(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        # Content attention: matrix_ac = q_u @ k^T
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -> (batch, heads, seq, seq)
        matrix_ac = mx.matmul(q_with_bias_u, k.transpose(0, 1, 3, 2))

        # Position attention: matrix_bd = q_v @ p^T
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -> (batch, heads, seq, seq)
        matrix_bd = mx.matmul(q_with_bias_v, p.transpose(0, 1, 3, 2))

        # NOTE: rel_shift is NOT used in original PyTorch (commented out)
        # matrix_bd = self._rel_shift(matrix_bd)

        # Combine and scale
        scores = (matrix_ac + matrix_bd) * self.scale

        # Apply mask
        if mask is not None and mask.size > 0:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
            scores = mx.where(mask, scores, mx.array(float("-inf")))

        attn = mx.softmax(scores, axis=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        # Output: (batch, heads, seq, head_dim)
        out = mx.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, seq_len, self.dim)
        out = self.linear_out(out)

        return out, cache


class Conv2dSubsampling2(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Matches original PyTorch structure:
    - self.conv = nn.Sequential(nn.Conv2d(1, odim, 3, 2), nn.ReLU())
    - self.out = nn.Sequential(nn.Linear(...))
    - self.pos_enc = pos_enc_class
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        pos_enc: Optional[nn.Module] = None,
    ):
        super().__init__()
        # Single Conv2d with stride=2 (matches original: nn.Conv2d(1, odim, 3, 2))
        # MLX Conv2d uses NHWC format
        # Original PyTorch: (batch, 1, time, mel_dim) NCHW -> (batch, odim, t', f')
        # MLX: (batch, time, mel_dim, 1) NHWC -> (batch, t', f', odim)
        self.conv = nn.Conv2d(1, output_dim, kernel_size=3, stride=2)

        # Output Linear: odim * ((idim - 1) // 2) -> odim
        # After conv with kernel=3, stride=2, no padding:
        # output_freq = (input_dim - 3) // 2 + 1 = (input_dim - 1) // 2
        self.out = nn.Linear(output_dim * ((input_dim - 1) // 2), output_dim)
        self.pos_enc = pos_enc

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input (batch, time, mel_dim)
            mask: Mask (batch, 1, time)

        Returns:
            Tuple of (output, pos_emb, new_mask)
        """
        # Add channel dim at the end for NHWC: (batch, time, mel_dim, 1)
        x = x[:, :, :, None]

        # Conv2d with stride=2: (batch, time, mel_dim, 1) -> (batch, t', f', odim)
        x = self.conv(x)
        x = nn.relu(x)

        # x is now (batch, t', f', output_dim) in NHWC format
        b, t, f, c = x.shape
        # PyTorch flattens as (c, f) but MLX has (f, c)
        # Need to transpose (f, c) -> (c, f) before flatten
        x = x.transpose(0, 1, 3, 2)  # (b, t, f, c) -> (b, t, c, f)
        x = x.reshape(b, t, c * f)   # (b, t, c*f)

        # Linear projection
        x = self.out(x)

        # Update mask: original uses mask[:, :, 2::2]
        if mask is not None and mask.size > 0:
            new_mask = mask[:, :, 2::2]
        else:
            new_mask = mask

        # Apply positional encoding
        pos_emb = None
        if self.pos_enc is not None:
            x, pos_emb = self.pos_enc(x)

        return x, pos_emb, new_mask


class ConformerEncoderLayer(nn.Module):
    """Single Conformer encoder layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        kernel_size: int = 15,
        normalize_before: bool = True,
        use_macaron: bool = False,
        use_cnn_module: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.normalize_before = normalize_before
        self.use_cnn_module = use_cnn_module

        # Feed-forward
        self.norm_ff = nn.LayerNorm(dim)
        self.feed_forward = PositionwiseFeedForward(dim, ff_dim, dropout)

        # Macaron-style second feed-forward
        self.ff_scale = 0.5 if use_macaron else 1.0
        if use_macaron:
            self.norm_ff_macaron = nn.LayerNorm(dim)
            self.feed_forward_macaron = PositionwiseFeedForward(dim, ff_dim, dropout)

        # Self-attention
        self.norm_mha = nn.LayerNorm(dim)
        self.self_attn = RelPositionMultiHeadAttention(num_heads, dim, dropout)

        # Convolution module
        if use_cnn_module:
            self.norm_conv = nn.LayerNorm(dim)
            self.conv_module = ConvolutionModule(dim, kernel_size, dropout)
            self.norm_final = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        pos_emb: mx.array,
        mask_pad: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, None, None]:
        """Forward pass.

        Args:
            x: Input (batch, seq_len, dim)
            mask: Attention mask
            pos_emb: Position embedding
            mask_pad: Padding mask for conv

        Returns:
            Tuple of (output, mask, None, None)
        """
        # Macaron feed-forward
        if hasattr(self, "feed_forward_macaron"):
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.feed_forward_macaron(x)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # Self-attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, _ = self.self_attn(x, x, x, mask, pos_emb)
        if self.dropout is not None:
            x_att = self.dropout(x_att)
        x = residual + x_att
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Convolution module
        if self.use_cnn_module:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x_conv = self.conv_module(x, mask_pad)
            if self.dropout is not None:
                x_conv = self.dropout(x_conv)
            x = residual + x_conv
            if not self.normalize_before:
                x = self.norm_conv(x)

        # Feed-forward
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x_ff = self.feed_forward(x)
        if self.dropout is not None:
            x_ff = self.dropout(x_ff)
        x = residual + self.ff_scale * x_ff
        if not self.normalize_before:
            x = self.norm_ff(x)

        # Final norm for conformer
        if self.use_cnn_module:
            x = self.norm_final(x)

        return x, mask, None, None


class ConformerEncoder(nn.Module):
    """Conformer encoder for audio conditioning."""

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.config = config

        # Input embedding with subsampling
        pos_enc = RelPositionalEncoding(config.output_size, config.dropout_rate)
        self.embed = Conv2dSubsampling2(
            config.input_size,
            config.output_size,
            config.dropout_rate,
            pos_enc,
        )

        # Encoder layers
        self.encoders = [
            ConformerEncoderLayer(
                dim=config.output_size,
                num_heads=config.attention_heads,
                ff_dim=config.linear_units,
                dropout=config.dropout_rate,
                kernel_size=config.cnn_module_kernel,
                normalize_before=config.normalize_before,
                use_cnn_module=config.use_cnn_module,
            )
            for _ in range(config.num_blocks)
        ]

        self.after_norm = nn.LayerNorm(config.output_size)

    def __call__(
        self,
        x: mx.array,
        x_lens: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            x: Input mel spectrogram (batch, mel_dim, time) or (batch, time, mel_dim)

        Returns:
            Tuple of (encoded features, mask)
        """
        # Handle different input formats
        if x.ndim == 3:
            # If (batch, mel_dim, time), transpose to (batch, time, mel_dim)
            if x.shape[1] == self.config.input_size:
                x = x.transpose(0, 2, 1)

        batch_size, time, _ = x.shape

        # Create mask
        if x_lens is not None:
            # Create mask from lengths
            mask = mx.arange(time)[None, :] < x_lens[:, None]
            mask = mask[:, None, :]  # (batch, 1, time)
        else:
            mask = mx.ones((batch_size, 1, time), dtype=mx.bool_)

        # Embedding with subsampling
        x, pos_emb, mask = self.embed(x, mask)

        # Create attention mask (all-ones for now, could be causal)
        chunk_mask = mask

        # Encoder layers
        for layer in self.encoders:
            x, chunk_mask, _, _ = layer(x, chunk_mask, pos_emb, mask)

        # Final normalization
        x = self.after_norm(x)

        return x, mask
