"""GPT-2 model implementation for IndexTTS."""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class GPT2Attention(nn.Module):
    """GPT-2 style multi-head attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection (HuggingFace GPT2 uses bias)
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Attention mask
            cache: KV cache tuple

        Returns:
            Output tensor and updated cache
        """
        batch_size, seq_len, _ = x.shape

        # Combined QKV projection
        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

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

        # Attention scores
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        # Apply mask
        if mask is not None:
            scores = scores + mask

        # Softmax
        attn = mx.softmax(scores, axis=-1)

        # Apply to values
        out = mx.matmul(attn, v)

        # Reshape back
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.dim)

        # Output projection
        out = self.c_proj(out)

        return out, new_cache


class GPT2MLP(nn.Module):
    """GPT-2 style MLP."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.c_fc = nn.Linear(dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass."""
        x = self.c_fc(x)
        x = nn.gelu_approx(x)  # GPT-2 uses GELU
        x = self.c_proj(x)
        return x


class GPT2Block(nn.Module):
    """GPT-2 transformer block."""

    def __init__(self, dim: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()

        self.ln_1 = nn.LayerNorm(dim)
        self.attn = GPT2Attention(dim, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = GPT2MLP(dim)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        """Forward pass."""
        # Self-attention with residual
        residual = x
        x = self.ln_1(x)
        x, new_cache = self.attn(x, mask, cache)
        x = residual + x

        # MLP with residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_cache


class GPT2Model(nn.Module):
    """GPT-2 model backbone.

    This is the transformer backbone without embeddings or output heads,
    as those are handled by the UnifiedVoice model.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int = 2048,
    ):
        """Initialize GPT-2 model.

        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Transformer blocks
        self.h = [
            GPT2Block(dim, num_heads, max_seq_len)
            for _ in range(num_layers)
        ]

        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)

    def __call__(
        self,
        inputs_embeds: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, Optional[List[Tuple[mx.array, mx.array]]]]:
        """Forward pass.

        Args:
            inputs_embeds: Input embeddings (batch, seq_len, dim)
            mask: Attention mask (if None, uses causal mask)
            cache: List of KV caches per layer

        Returns:
            Output hidden states and updated cache
        """
        x = inputs_embeds
        query_len = x.shape[1]
        new_cache = []

        # Calculate key length (includes cached tokens if any)
        if cache is not None and cache[0] is not None:
            cache_len = cache[0][0].shape[1]  # K cache shape: (batch, cache_len, dim)
            key_len = cache_len + query_len
        else:
            key_len = query_len

        # Create causal mask if not provided (GPT-2 is autoregressive)
        if mask is None:
            mask = self.create_causal_mask(query_len, key_len)

        for i, block in enumerate(self.h):
            layer_cache = cache[i] if cache is not None else None
            x, updated_cache = block(x, mask, layer_cache)
            new_cache.append(updated_cache)

        x = self.ln_f(x)

        return x, new_cache

    def create_causal_mask(self, query_len: int, key_len: int) -> mx.array:
        """Create causal attention mask.

        For autoregressive generation with KV cache:
        - query_len: length of current input (1 during generation)
        - key_len: total length including cached keys (cache_len + query_len)

        Args:
            query_len: Query sequence length
            key_len: Key sequence length (may be larger due to cache)

        Returns:
            Causal mask of shape (1, 1, query_len, key_len)
            Where positions that should NOT be attended have -inf
        """
        # Create mask where each query position can only attend to
        # key positions up to and including itself
        # For incremental decoding: query at position i attends to keys 0..i
        #
        # With cache, if we're at step N (cache has N tokens, query_len=1):
        # - key_len = N + 1
        # - We want mask shape (1, key_len) = (1, N+1) all zeros (can attend to all)
        #
        # Without cache (initial forward, query_len = key_len = S):
        # - Standard lower triangular mask
        if query_len == key_len:
            # Standard causal mask for initial forward pass
            mask = mx.triu(mx.ones((query_len, key_len)), k=1)
        else:
            # Incremental decoding: query can attend to all keys
            # (causal constraint is already satisfied by only having past keys in cache)
            mask = mx.zeros((query_len, key_len))

        # Convert to additive mask: 0 -> 0, 1 -> -inf
        mask = mx.where(mask > 0, float("-inf"), 0.0)
        return mask[None, None, :, :]
