"""Attention modules for IndexTTS."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class LearnedPositionEmbedding(nn.Module):
    """Learned position embeddings for GPT-style models.

    Original structure: nn.Embedding(seq_len, model_dim) named 'emb'
    """

    def __init__(self, max_seq_len: int, dim: int, init_std: float = 0.02):
        """Initialize position embedding.

        Args:
            max_seq_len: Maximum sequence length
            dim: Embedding dimension
            init_std: Standard deviation for initialization
        """
        super().__init__()
        # Named 'emb' to match original PyTorch: self.emb = nn.Embedding(seq_len, model_dim)
        self.emb = nn.Embedding(max_seq_len, dim)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """Get position embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (batch, seq_len)
            offset: Position offset for incremental decoding

        Returns:
            Position embeddings of shape (seq_len, dim) or (1, dim) if offset given
        """
        if isinstance(offset, int) and offset > 0:
            # Single position for incremental decoding
            positions = mx.array([offset])
            return self.emb(positions)

        seq_len = x.shape[1] if x.ndim >= 2 else x.shape[0]
        positions = mx.arange(seq_len)
        return self.emb(positions)

    def get_fixed_embedding(self, position: int, device=None) -> mx.array:
        """Get embedding for a single position.

        Args:
            position: Position index
            device: Ignored (for PyTorch compatibility)

        Returns:
            Position embedding of shape (1, 1, dim)
        """
        pos = mx.array([position])
        return self.emb(pos)[None, :]


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Conformer."""

    def __init__(self, dim: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Pre-compute positional encodings
        self._pe = None

    def _compute_pe(self, max_len: int) -> mx.array:
        """Compute sinusoidal positional encoding."""
        pe = mx.zeros((max_len, self.dim))
        position = mx.arange(0, max_len, dtype=mx.float32)[:, None]
        div_term = mx.exp(mx.arange(0, self.dim, 2, dtype=mx.float32) * -(math.log(10000.0) / self.dim))

        # Use numpy for initial computation, then convert
        import numpy as np
        pos_np = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_np = np.exp(np.arange(0, self.dim, 2, dtype=np.float32) * -(np.log(10000.0) / self.dim))

        pe_np = np.zeros((max_len, self.dim), dtype=np.float32)
        pe_np[:, 0::2] = np.sin(pos_np * div_np)
        pe_np[:, 1::2] = np.cos(pos_np * div_np)

        return mx.array(pe_np)

    @property
    def pe(self) -> mx.array:
        if self._pe is None:
            self._pe = self._compute_pe(self.max_len)
        return self._pe

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Apply positional encoding.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)

        Returns:
            Tuple of (x + pe, pe) where pe has shape (1, seq_len, dim)
        """
        seq_len = x.shape[1]
        pe = self.pe[:seq_len][None, :, :]  # (1, seq_len, dim)

        x = x + pe
        if self.dropout is not None:
            x = self.dropout(x)

        return x, pe


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            query: Query tensor (batch, seq_len, dim)
            key: Key tensor (batch, seq_len, dim)
            value: Value tensor (batch, seq_len, dim)
            mask: Attention mask
            cache: KV cache tuple

        Returns:
            Output tensor and updated cache
        """
        batch_size, seq_len, _ = query.shape

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Handle cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)

        new_cache = (k, v)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Compute attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        # Apply mask
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        attn_weights = mx.softmax(scores, axis=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = mx.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.dim)

        # Output projection
        output = self.out_proj(output)

        return output, new_cache


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding for Conformer."""

    def __init__(
        self,
        num_heads: int,
        dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim)
        self.linear_k = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, dim)
        self.linear_out = nn.Linear(dim, dim)

        # Positional encoding projections
        self.linear_pos = nn.Linear(dim, dim, bias=False)

        # Bias parameters for relative position
        self.pos_bias_u = mx.zeros((self.num_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((self.num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _rel_shift(self, x: mx.array) -> mx.array:
        """Compute relative positional encoding shift.

        Args:
            x: Input tensor (batch, heads, seq_len, 2*seq_len-1)

        Returns:
            Shifted tensor (batch, heads, seq_len, seq_len)
        """
        batch_size, num_heads, qlen, pos_len = x.shape

        # Pad and reshape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = x.reshape(batch_size, num_heads, pos_len + 1, qlen)
        x = x[:, :, 1:, :]
        x = x.reshape(batch_size, num_heads, qlen, pos_len)

        # Slice to get correct size
        x = x[:, :, :, :qlen]

        return x

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

        Args:
            query: Query tensor (batch, seq_len, dim)
            key: Key tensor (batch, seq_len, dim)
            value: Value tensor (batch, seq_len, dim)
            mask: Attention mask (batch, 1, seq_len) or (batch, seq_len, seq_len)
            pos_emb: Positional embedding (1, seq_len, dim)
            cache: KV cache

        Returns:
            Output tensor and cache
        """
        batch_size, seq_len, _ = query.shape

        # Linear projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose: (batch, heads, seq_len, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Add position bias
        q_with_bias_u = q + self.pos_bias_u[None, :, None, :]
        q_with_bias_v = q + self.pos_bias_v[None, :, None, :]

        # Content-based attention
        matrix_ac = mx.matmul(q_with_bias_u, k.transpose(0, 1, 3, 2))

        # Position-based attention
        p = self.linear_pos(pos_emb)
        p = p.reshape(1, -1, self.num_heads, self.head_dim)
        p = p.transpose(0, 2, 1, 3)

        matrix_bd = mx.matmul(q_with_bias_v, p.transpose(0, 1, 3, 2))
        matrix_bd = self._rel_shift(matrix_bd)

        # Combine scores
        scores = (matrix_ac + matrix_bd) * self.scale

        # Apply mask
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
            scores = mx.where(mask, scores, mx.array(float("-inf")))

        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply to values
        output = mx.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)

        # Output projection
        output = self.linear_out(output)

        return output, cache


class AttentionBlock(nn.Module):
    """Attention block used in conditioning encoder."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, dim, seq_len) - note: channel-first!

        Returns:
            Output tensor (batch, dim, seq_len)
        """
        # Transpose to (batch, seq_len, dim)
        x = x.transpose(0, 2, 1)

        # Self-attention with residual
        residual = x
        x = self.norm(x)
        x, _ = self.attention(x, x, x)
        x = residual + x

        # Transpose back to (batch, dim, seq_len)
        x = x.transpose(0, 2, 1)

        return x
