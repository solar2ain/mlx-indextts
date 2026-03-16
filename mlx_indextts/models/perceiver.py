"""Perceiver Resampler for IndexTTS conditioning."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """RMS Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))  # Called 'gamma' in original but renamed by converter

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization."""
        # RMS norm: x / sqrt(mean(x^2) + eps) * weight
        rms = mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * self.weight


class GEGLU(nn.Module):
    """GEGLU activation: x * GELU(gate)."""

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.gelu(gate)


class PerceiverFeedForward(nn.Module):
    """Feed-forward network with GEGLU activation.

    Structure matches original: Sequential(Linear, GEGLU, Linear)
    Parameter names: .0.weight, .0.bias (first Linear), .2.weight, .2.bias (second Linear)

    Converted to: w_1.weight, w_1.bias (first), w_2.weight, w_2.bias (second)
    """

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        # Using numbered attributes to match original Sequential structure
        # Original: Sequential(Linear, GEGLU, Linear) -> indices .0, .1, .2
        # Our converter maps .0 -> w_1, .2 -> w_2
        self.w_1 = nn.Linear(dim, inner_dim * 2)  # First linear (before GEGLU)
        self.w_2 = nn.Linear(inner_dim, dim)       # Second linear (after GEGLU)
        self._activation = GEGLU()
        self._dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        x = self.w_1(x)
        x = self._activation(x)
        if self._dropout is not None:
            x = self._dropout(x)
        x = self.w_2(x)
        return x


class PerceiverAttention(nn.Module):
    """Cross-attention for Perceiver Resampler.

    Queries attend to both themselves and the context.

    Original structure:
    - to_q: Linear(dim, inner_dim, bias=False)
    - to_kv: Linear(dim_context, inner_dim * 2, bias=False)  # split into K and V
    - to_out: Linear(inner_dim, dim, bias=False)

    Converter splits to_kv into linear_k and linear_v
    """

    def __init__(
        self,
        dim: int,
        dim_context: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        dim_context = dim_context or dim
        inner_dim = num_heads * head_dim

        # These names match what converter produces
        self.linear_q = nn.Linear(dim, inner_dim, bias=False)
        self.linear_k = nn.Linear(dim_context, inner_dim, bias=False)
        self.linear_v = nn.Linear(dim_context, inner_dim, bias=False)
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

        self._dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        x: mx.array,
        context: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Query tensor (batch, n_latents, dim)
            context: Key/value tensor (batch, seq_len, dim_context)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, n_latents, dim)
        """
        batch_size, n_latents, _ = x.shape

        # Concatenate x and context for keys/values
        # This allows latents to attend to both themselves and context
        kv_input = mx.concatenate([x, context], axis=1)

        # Project Q, K, V
        q = self.linear_q(x)
        k = self.linear_k(kv_input)
        v = self.linear_v(kv_input)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, n_latents, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose: (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Compute attention
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        # Apply mask if provided
        if mask is not None:
            # Expand mask for latents (they can always attend)
            latent_mask = mx.ones((batch_size, n_latents), dtype=mx.bool_)
            full_mask = mx.concatenate([latent_mask, mask], axis=1)
            full_mask = full_mask[:, None, None, :]  # (batch, 1, 1, total_seq)
            scores = mx.where(full_mask, scores, mx.array(float("-inf")))

        attn = mx.softmax(scores, axis=-1)
        if self._dropout is not None:
            attn = self._dropout(attn)

        # Apply attention to values
        out = mx.matmul(attn, v)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, n_latents, -1)

        return self.linear_out(out)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler for conditioning.

    Compresses variable-length audio features into fixed number of latents.

    Original structure uses ModuleList of [Attention, FeedForward] pairs.
    Parameter paths:
    - layers.0.0.* -> first layer attention
    - layers.0.1.* -> first layer feedforward
    - layers.1.0.* -> second layer attention
    - layers.1.1.* -> second layer feedforward
    """

    def __init__(
        self,
        dim: int,
        n_dim_context: Optional[int] = None,
        n_latents: int = 32,
        n_heads: int = 8,
        n_head_dim: int = 64,
        n_ff_mult: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize Perceiver Resampler.

        Args:
            dim: Output dimension
            n_dim_context: Context dimension (default: same as dim)
            n_latents: Number of output latents
            n_heads: Number of attention heads
            n_head_dim: Dimension per head
            n_ff_mult: Feed-forward multiplier
            n_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        n_dim_context = n_dim_context or dim

        # Project context if dimensions differ
        if n_dim_context != dim:
            self.proj_context = nn.Linear(n_dim_context, dim)
        else:
            self.proj_context = None

        # Learnable latent queries
        self.latents = mx.zeros((n_latents, dim))

        # Build layers as list of [attention, ff] pairs to match original structure
        # This creates parameter paths: layers.i.0.* (attention), layers.i.1.* (ff)
        self.layers = []
        for _ in range(n_layers):
            attn = PerceiverAttention(
                dim=dim,
                dim_context=dim,  # After projection
                num_heads=n_heads,
                head_dim=n_head_dim,
                dropout=dropout,
            )
            ff = PerceiverFeedForward(dim, mult=n_ff_mult, dropout=dropout)
            self.layers.append([attn, ff])

        # Output normalization
        self.norm = RMSNorm(dim)

    def __call__(
        self,
        context: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            context: Encoder output (batch, seq_len, dim_context)
            mask: Optional padding mask (batch, seq_len)

        Returns:
            Latent features (batch, n_latents, dim)
        """
        batch_size = context.shape[0]

        # Project context if needed
        if self.proj_context is not None:
            context = self.proj_context(context)

        # Expand latents for batch
        latents = mx.broadcast_to(
            self.latents[None, :, :],
            (batch_size, self.latents.shape[0], self.latents.shape[1])
        )
        # Need to make it a new array to allow gradients
        latents = latents + mx.zeros_like(latents)

        # Apply perceiver layers
        for attn, ff in self.layers:
            latents = latents + attn(latents, context, mask)
            latents = latents + ff(latents)

        # Final normalization
        latents = self.norm(latents)

        return latents
