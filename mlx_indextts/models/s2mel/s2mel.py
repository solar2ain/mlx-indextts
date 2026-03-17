"""S2Mel module - complete Semantic-to-Mel model.

This module implements the full S2Mel pipeline including:
- GPT layer projection (semantic -> content)
- Length regulator (interpolation to mel length)
- CFM (conditional flow matching for mel generation)
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_indextts.models.s2mel.length_regulator import InterpolateRegulator
from mlx_indextts.models.s2mel.cfm import CFM


class GPTLayer(nn.Module):
    """Simple MLP to project GPT output to content dimension.

    Architecture: Linear(1280, 256) -> Linear(256, 128) -> Linear(128, 1024)
    """

    def __init__(self, in_dim: int = 1280, hidden_dims: tuple = (256, 128), out_dim: int = 1024):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.layers = [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1)]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, in_dim)

        Returns:
            Output tensor (batch, seq_len, out_dim)
        """
        for layer in self.layers:
            x = layer(x)
        return x


class S2Mel(nn.Module):
    """Full Semantic-to-Mel model.

    This combines:
    1. gpt_layer: Projects GPT output to content dimension
    2. length_regulator: Upsamples content to mel length
    3. cfm: Generates mel spectrogram via diffusion
    """

    def __init__(
        self,
        # GPT layer config
        gpt_in_dim: int = 1280,
        gpt_hidden_dims: tuple = (256, 128),
        gpt_out_dim: int = 1024,
        # Length regulator config
        lr_channels: int = 512,
        lr_sampling_ratios: tuple = (1, 1, 1, 1),
        lr_is_discrete: bool = False,
        lr_in_channels: int = 1024,
        lr_codebook_size: int = 2048,
        lr_n_codebooks: int = 1,
        # CFM config
        cfm_in_channels: int = 80,
        cfm_hidden_dim: int = 512,
        cfm_num_heads: int = 8,
        cfm_depth: int = 13,
        cfm_content_dim: int = 512,
        cfm_style_dim: int = 192,
        cfm_class_dropout_prob: float = 0.1,
        cfm_long_skip_connection: bool = True,
        cfm_uvit_skip_connection: bool = True,
        cfm_time_as_token: bool = False,
        cfm_style_as_token: bool = False,
        cfm_style_condition: bool = True,
        cfm_final_layer_type: str = 'wavenet',
        cfm_wavenet_hidden_dim: int = 512,
        cfm_wavenet_num_layers: int = 8,
        cfm_wavenet_kernel_size: int = 5,
        cfm_wavenet_dilation_rate: int = 1,
        cfm_wavenet_p_dropout: float = 0.2,
        cfm_zero_prompt_speech_token: bool = False,
    ):
        super().__init__()

        # GPT layer projection
        self.gpt_layer = GPTLayer(
            in_dim=gpt_in_dim,
            hidden_dims=gpt_hidden_dims,
            out_dim=gpt_out_dim,
        )

        # Length regulator
        self.length_regulator = InterpolateRegulator(
            channels=lr_channels,
            sampling_ratios=lr_sampling_ratios,
            is_discrete=lr_is_discrete,
            in_channels=lr_in_channels,
            codebook_size=lr_codebook_size,
            n_codebooks=lr_n_codebooks,
        )

        # CFM (Conditional Flow Matching)
        self.cfm = CFM(
            in_channels=cfm_in_channels,
            hidden_dim=cfm_hidden_dim,
            num_heads=cfm_num_heads,
            depth=cfm_depth,
            content_dim=cfm_content_dim,
            style_dim=cfm_style_dim,
            class_dropout_prob=cfm_class_dropout_prob,
            long_skip_connection=cfm_long_skip_connection,
            uvit_skip_connection=cfm_uvit_skip_connection,
            time_as_token=cfm_time_as_token,
            style_as_token=cfm_style_as_token,
            style_condition=cfm_style_condition,
            final_layer_type=cfm_final_layer_type,
            wavenet_hidden_dim=cfm_wavenet_hidden_dim,
            wavenet_num_layers=cfm_wavenet_num_layers,
            wavenet_kernel_size=cfm_wavenet_kernel_size,
            wavenet_dilation_rate=cfm_wavenet_dilation_rate,
            wavenet_p_dropout=cfm_wavenet_p_dropout,
            zero_prompt_speech_token=cfm_zero_prompt_speech_token,
        )

    def inference(
        self,
        gpt_hidden: mx.array,
        mel_lens: mx.array,
        prompt_mel: mx.array,
        style: mx.array,
        n_timesteps: int = 32,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.7,
    ) -> mx.array:
        """Full S2Mel inference pipeline.

        Args:
            gpt_hidden: GPT hidden states (batch, seq_len, gpt_dim)
            mel_lens: Target mel lengths (batch,)
            prompt_mel: Reference mel (batch, 80, prompt_len)
            style: Style embedding (batch, style_dim)
            n_timesteps: Number of diffusion steps
            temperature: Noise temperature
            inference_cfg_rate: CFG rate

        Returns:
            Generated mel spectrogram (batch, 80, mel_len)
        """
        # 1. Project GPT hidden to content dimension
        content = self.gpt_layer(gpt_hidden)  # (batch, seq_len, 1024)

        # 2. Upsample to mel length via length regulator
        mu, _, _, _, _ = self.length_regulator(content, mel_lens, n_quantizers=3)
        # mu: (batch, mel_len, 512)

        # 3. Generate mel via CFM diffusion
        mel = self.cfm.inference(
            mu=mu,
            x_lens=mel_lens,
            prompt=prompt_mel,
            style=style,
            f0=None,
            n_timesteps=n_timesteps,
            temperature=temperature,
            inference_cfg_rate=inference_cfg_rate,
        )

        return mel


def create_s2mel_from_config(config) -> S2Mel:
    """Create S2Mel model from config dictionary.

    Args:
        config: S2Mel config section from IndexTTS config

    Returns:
        S2Mel model
    """
    dit_config = config.get('DiT', {})
    wavenet_config = config.get('wavenet', {})
    style_config = config.get('style_encoder', {})
    lr_config = config.get('length_regulator', {})

    return S2Mel(
        # GPT layer config
        gpt_in_dim=config.get('gpt_in_dim', 1280),
        gpt_hidden_dims=tuple(config.get('gpt_hidden_dims', [256, 128])),
        gpt_out_dim=config.get('gpt_out_dim', 1024),
        # Length regulator config
        lr_channels=lr_config.get('channels', 512),
        lr_sampling_ratios=tuple(lr_config.get('sampling_ratios', [1, 1, 1, 1])),
        lr_is_discrete=lr_config.get('is_discrete', False),
        lr_in_channels=lr_config.get('in_channels', 1024),
        lr_codebook_size=lr_config.get('codebook_size', 2048),
        lr_n_codebooks=lr_config.get('n_codebooks', 1),
        # CFM config
        cfm_in_channels=dit_config.get('in_channels', 80),
        cfm_hidden_dim=dit_config.get('hidden_dim', 512),
        cfm_num_heads=dit_config.get('num_heads', 8),
        cfm_depth=dit_config.get('depth', 13),
        cfm_content_dim=dit_config.get('content_dim', 512),
        cfm_style_dim=style_config.get('dim', 192),
        cfm_class_dropout_prob=dit_config.get('class_dropout_prob', 0.1),
        cfm_long_skip_connection=dit_config.get('long_skip_connection', True),
        cfm_uvit_skip_connection=dit_config.get('uvit_skip_connection', True),
        cfm_time_as_token=dit_config.get('time_as_token', False),
        cfm_style_as_token=dit_config.get('style_as_token', False),
        cfm_style_condition=dit_config.get('style_condition', True),
        cfm_final_layer_type=dit_config.get('final_layer_type', 'wavenet'),
        cfm_wavenet_hidden_dim=wavenet_config.get('hidden_dim', 512),
        cfm_wavenet_num_layers=wavenet_config.get('num_layers', 8),
        cfm_wavenet_kernel_size=wavenet_config.get('kernel_size', 5),
        cfm_wavenet_dilation_rate=wavenet_config.get('dilation_rate', 1),
        cfm_wavenet_p_dropout=wavenet_config.get('p_dropout', 0.2),
        cfm_zero_prompt_speech_token=dit_config.get('zero_prompt_speech_token', False),
    )
