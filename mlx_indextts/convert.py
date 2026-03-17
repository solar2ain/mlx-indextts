"""Model conversion from PyTorch to MLX."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_indextts.config import IndexTTSConfig


def load_pytorch_weights(checkpoint_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load PyTorch checkpoint and convert to numpy arrays.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pth file)

    Returns:
        Dictionary of numpy arrays
    """
    import torch

    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Convert to numpy
    numpy_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert bfloat16 to float32 first (numpy doesn't support bfloat16)
            if value.dtype == torch.bfloat16:
                value = value.float()
            numpy_dict[key] = value.cpu().numpy()

    return numpy_dict


def convert_gpt_weights(weights: Dict[str, np.ndarray], config: IndexTTSConfig) -> Dict[str, mx.array]:
    """Convert GPT weights from PyTorch to MLX format.

    Args:
        weights: PyTorch weights as numpy arrays
        config: Model configuration

    Returns:
        MLX-compatible weights
    """
    new_weights = {}
    num_layers = config.gpt.layers

    for key, value in weights.items():
        new_key = key

        # Skip certain keys
        if "num_batches_tracked" in key:
            continue
        # Skip causal mask buffer (HF GPT2 stores it as 'attn.bias', but NOT c_attn.bias)
        if key.endswith("attn.bias") and "c_attn" not in key and "c_proj" not in key:
            continue
        # Skip pe (positional encoding) - we compute it dynamically
        if ".pos_enc.pe" in key or (".pe" in key and "perceiver" not in key):
            continue

        # Handle conditioning_encoder.embed naming differences
        # PyTorch: embed.conv.0 (single Conv2d in Sequential) -> MLX: embed.conv
        # PyTorch: embed.out.0 (Linear in Sequential) -> MLX: embed.out
        # Works for both conditioning_encoder and emo_conditioning_encoder
        if "conditioning_encoder.embed.conv.0." in key:
            new_key = new_key.replace("embed.conv.0.", "embed.conv.")
        if "conditioning_encoder.embed.out.0." in key:
            new_key = new_key.replace("embed.out.0.", "embed.out.")

        # Handle conv layers (transpose for MLX)
        # PyTorch Conv1d: (out, in, k) -> MLX: (out, k, in)
        # PyTorch Conv2d: (out, in, kH, kW) -> MLX: (out, kH, kW, in)
        if ("conv" in key or "init" in key or "cond" in key) and "weight" in key:
            if value.ndim == 3:
                value = value.transpose(0, 2, 1)
            elif value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)

        # Handle GPT-2 attention/MLP weight transposition
        for i in range(num_layers):
            if f"gpt.h.{i}.attn.c_attn.weight" in key:
                value = value.T
            elif f"gpt.h.{i}.attn.c_proj.weight" in key:
                value = value.T
            elif f"gpt.h.{i}.mlp.c_fc.weight" in key:
                value = value.T
            elif f"gpt.h.{i}.mlp.c_proj.weight" in key:
                value = value.T

        # Handle perceiver weights (both spk and emo)
        if "perceiver_encoder" in key or "emo_perceiver_encoder" in key:
            # to_q -> linear_q
            new_key = new_key.replace("to_q.", "linear_q.")
            # to_out -> linear_out
            new_key = new_key.replace("to_out.", "linear_out.")

            # Handle to_kv split
            if "to_kv.weight" in key:
                # Split into k and v
                k_weight, v_weight = np.split(value, 2, axis=0)
                k_key = new_key.replace("to_kv.weight", "linear_k.weight")
                v_key = new_key.replace("to_kv.weight", "linear_v.weight")
                new_weights[k_key] = mx.array(k_weight)
                new_weights[v_key] = mx.array(v_weight)
                continue

            # Handle FF layers in Perceiver
            # Original: layers.X.1.0.weight (first Linear in Sequential FF)
            # Original: layers.X.1.2.weight (second Linear in Sequential FF)
            # Target:   layers.X.1.w_1.weight / layers.X.1.w_2.weight
            # Use regex to be more precise (only match .1.0. or .1.2. after layers.N)
            import re
            new_key = re.sub(r"\.(\d+)\.1\.0\.", r".\1.1.w_1.", new_key)
            new_key = re.sub(r"\.(\d+)\.1\.2\.", r".\1.1.w_2.", new_key)

            # norm.gamma -> norm.weight
            new_key = new_key.replace("norm.gamma", "norm.weight")

        new_weights[new_key] = mx.array(value)

    return new_weights


def convert_bigvgan_weights(weights: Dict[str, np.ndarray], config: IndexTTSConfig) -> Dict[str, mx.array]:
    """Convert BigVGAN weights from PyTorch to MLX format.

    Handles:
    - Weight normalization (weight_g, weight_v -> weight)
    - Alias-free activation filters (skip, computed at runtime)
    - Snake/SnakeBeta activation parameters (alpha, beta)

    Args:
        weights: PyTorch weights as numpy arrays
        config: Model configuration

    Returns:
        MLX-compatible weights
    """
    new_weights = {}

    # First pass: collect weight_g and weight_v pairs
    weight_v_keys = {}
    weight_g_keys = {}
    for key in weights.keys():
        if key.endswith(".weight_v"):
            base_key = key[:-9]  # Remove ".weight_v"
            weight_v_keys[base_key] = key
        elif key.endswith(".weight_g"):
            base_key = key[:-9]  # Remove ".weight_g"
            weight_g_keys[base_key] = key

    # Process weights
    processed_keys = set()

    for key, value in weights.items():
        # Skip batch norm tracking
        if "num_batches_tracked" in key:
            continue

        # Skip alias-free filter weights (computed at runtime)
        if ".downsample." in key or ".upsample." in key:
            if "filter" in key:
                continue

        # Handle weight normalization - combine weight_g and weight_v
        if key.endswith(".weight_v"):
            base_key = key[:-9]
            if base_key in weight_g_keys:
                # Combine: weight = weight_g * (weight_v / ||weight_v||)
                weight_v = value
                weight_g = weights[weight_g_keys[base_key]]

                # Compute normalized weight
                # weight_v shape is (out_channels, in_channels, kernel_size) for Conv1d
                # norm is computed over all dimensions except the first
                norm_dims = tuple(range(1, weight_v.ndim))
                weight_v_norm = np.sqrt(np.sum(weight_v ** 2, axis=norm_dims, keepdims=True))
                weight = weight_g * (weight_v / (weight_v_norm + 1e-8))

                new_key = base_key + ".weight"
                processed_keys.add(key)
                processed_keys.add(weight_g_keys[base_key])
            else:
                # No matching weight_g, use as-is
                weight = value
                new_key = base_key + ".weight"
        elif key.endswith(".weight_g"):
            # Skip, handled with weight_v
            continue
        else:
            weight = value
            new_key = key

        # Handle naming differences
        new_key = (
            new_key.replace("norm.norm", "norm")
            .replace("conv.conv", "conv")
            .replace("conv1.conv", "conv1")
            .replace("conv2.conv", "conv2")
            .replace("fc.conv", "fc")
            .replace("asp_bn.norm", "asp_bn")
        )

        # Handle activation parameters (Snake/SnakeBeta)
        # With Activation1d wrapper, keep the .act. prefix
        # PyTorch: activation_post.act.alpha -> MLX: activation_post.act.alpha
        # PyTorch: resblocks.0.activations.0.act.alpha -> MLX: resblocks.0.activations.0.act.alpha
        # No changes needed to key - keep as is

        # Handle conv/transpose conv weight shapes
        if ("conv" in key or "cond" in key or "fc" in key) and weight.ndim >= 3:
            if weight.ndim == 3:
                # Conv1d: PyTorch (out, in, k) -> MLX (out, k, in)
                weight = weight.transpose(0, 2, 1)
            elif weight.ndim == 4:
                weight = weight.transpose(0, 2, 3, 1)

        # Handle upsampling transpose conv
        if "ups." in key and weight.ndim == 3:
            # ConvTranspose1d weights need different handling
            # PyTorch: (in, out, k) -> MLX: (out, k, in)
            weight = weight.transpose(1, 2, 0)

        new_weights[new_key] = mx.array(weight)

    return new_weights


def _quantize_weights(
    weights: Dict[str, mx.array],
    config: IndexTTSConfig,
    bits: int = 8,
    group_size: int = 64,
) -> Dict[str, mx.array]:
    """Quantize GPT weights for reduced memory usage.

    Only quantizes Linear layers in the GPT backbone (gpt.h.*) where dimensions
    are divisible by group_size. Other layers (embeddings, heads, perceiver)
    are kept in full precision.

    Args:
        weights: Original weights dictionary
        config: Model configuration
        bits: Quantization bits (4 or 8)
        group_size: Quantization group size

    Returns:
        Quantized weights dictionary
    """
    import re

    new_weights = {}
    quantized_count = 0
    skipped_count = 0

    for key, value in weights.items():
        # Only quantize GPT backbone linear layers (gpt.h.*.attn.* and gpt.h.*.mlp.*)
        # These are the largest components and benefit most from quantization
        is_gpt_backbone = bool(re.match(r"gpt\.h\.\d+\.(attn|mlp)\.", key))
        is_weight = key.endswith(".weight") and value.ndim == 2

        if is_gpt_backbone and is_weight:
            out_dim, in_dim = value.shape
            if in_dim % group_size == 0 and out_dim % group_size == 0:
                # Quantize this weight
                quantized, scales, biases = mx.quantize(value, bits=bits, group_size=group_size)
                # Store with special naming convention for quantized weights
                base_key = key[:-7]  # Remove ".weight"
                new_weights[f"{base_key}.weight"] = quantized
                new_weights[f"{base_key}.scales"] = scales
                new_weights[f"{base_key}.biases"] = biases
                quantized_count += 1
                continue
            else:
                skipped_count += 1

        new_weights[key] = value

    print(f"  Quantized {quantized_count} layers, kept {skipped_count} layers in fp32")
    return new_weights


def convert_model(
    model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    quantize_bits: Optional[int] = None,
) -> None:
    """Convert IndexTTS PyTorch model to MLX format.

    Args:
        model_dir: Directory containing PyTorch checkpoints
        output_dir: Output directory for MLX weights
        config_path: Optional path to config.yaml (default: model_dir/config.yaml)
        quantize_bits: Quantization bits (4, 8, or None for fp32). Default: None (fp32)
    """
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config_path is None:
        config_path = model_dir / "config.yaml"
    config = IndexTTSConfig.from_yaml(str(config_path))

    print(f"Converting IndexTTS model from {model_dir}")
    print(f"Version: {config.version or '1.0'}")
    if quantize_bits:
        print(f"Quantization: {quantize_bits}-bit")
    else:
        print(f"Quantization: None (fp32)")

    # Convert GPT weights
    gpt_path = model_dir / config.gpt_checkpoint
    if gpt_path.exists():
        print(f"Converting GPT weights from {gpt_path}")
        gpt_numpy = load_pytorch_weights(gpt_path)
        gpt_weights = convert_gpt_weights(gpt_numpy, config)

        # Apply quantization if requested
        if quantize_bits:
            print(f"Quantizing GPT to {quantize_bits}-bit...")
            gpt_weights = _quantize_weights(gpt_weights, config, quantize_bits)

        # Save GPT weights
        gpt_output = output_dir / "gpt.safetensors"
        mx.save_safetensors(str(gpt_output), gpt_weights)
        print(f"Saved GPT weights to {gpt_output}")
    else:
        print(f"Warning: GPT checkpoint not found at {gpt_path}")

    # Convert BigVGAN weights
    bigvgan_path = model_dir / config.bigvgan_checkpoint
    if bigvgan_path.exists():
        print(f"Converting BigVGAN weights from {bigvgan_path}")
        bigvgan_numpy = load_pytorch_weights(bigvgan_path)
        bigvgan_weights = convert_bigvgan_weights(bigvgan_numpy, config)

        # Save BigVGAN weights
        bigvgan_output = output_dir / "bigvgan.safetensors"
        mx.save_safetensors(str(bigvgan_output), bigvgan_weights)
        print(f"Saved BigVGAN weights to {bigvgan_output}")
    else:
        print(f"Warning: BigVGAN checkpoint not found at {bigvgan_path}")

    # Copy BPE model
    bpe_path = model_dir / config.bpe_model
    if bpe_path.exists():
        import shutil
        shutil.copy(bpe_path, output_dir / "tokenizer.model")
        print(f"Copied BPE model to {output_dir / 'tokenizer.model'}")

    # Save config as JSON (include quantization info)
    config_output = output_dir / "config.json"
    config_dict = config.to_dict()
    if quantize_bits:
        config_dict["quantize_bits"] = quantize_bits
    with open(config_output, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Saved config to {config_output}")

    print(f"\nConversion complete! Model saved to {output_dir}")


def load_mlx_model(
    model_dir: Union[str, Path],
) -> tuple:
    """Load converted MLX model.

    Args:
        model_dir: Directory containing converted MLX weights

    Returns:
        Tuple of (config, gpt_weights, bigvgan_weights, quantize_bits)
    """
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # Check if model is quantized
    quantize_bits = config_dict.get("quantize_bits", None)

    # Reconstruct config
    from mlx_indextts.config import GPTConfig, BigVGANConfig, MelConfig, ConformerConfig

    gpt_dict = config_dict.get("gpt", {})
    cond_module = gpt_dict.pop("condition_module", None)
    if cond_module:
        cond_module = ConformerConfig(**cond_module)

    config = IndexTTSConfig(
        gpt=GPTConfig(**gpt_dict, condition_module=cond_module),
        bigvgan=BigVGANConfig(**config_dict.get("bigvgan", {})),
        mel=MelConfig(**config_dict.get("mel", {})),
        bpe_model=config_dict.get("bpe_model", "tokenizer.model"),
        gpt_checkpoint=config_dict.get("gpt_checkpoint", "gpt.safetensors"),
        bigvgan_checkpoint=config_dict.get("bigvgan_checkpoint", "bigvgan.safetensors"),
        version=config_dict.get("version"),
        sample_rate=config_dict.get("sample_rate", 24000),
    )

    # Load weights
    gpt_weights = mx.load(str(model_dir / "gpt.safetensors"))
    bigvgan_weights = mx.load(str(model_dir / "bigvgan.safetensors"))

    return config, gpt_weights, bigvgan_weights, quantize_bits
