"""Model conversion from PyTorch to MLX for IndexTTS 2.0.

This module handles conversion of IndexTTS 2.0 models which have a different
architecture from 1.5:
- GPT v2: Extended with emotion conditioning
- S2Mel: New diffusion-based mel generation
- BigVGAN v2: Pure vocoder (no speaker encoder)

Usage (via CLI, auto-detects version):
    mlx-indextts convert --model-dir ~/Projects/index-tts/indexTTS-2 -o models/mlx-indexTTS-2.0
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import mlx.core as mx
import numpy as np

from mlx_indextts.config import IndexTTSConfig
from mlx_indextts.convert import load_pytorch_weights, convert_gpt_weights, _quantize_weights


def convert_gpt_v2_weights(weights: Dict[str, np.ndarray], config: IndexTTSConfig) -> Dict[str, mx.array]:
    """Convert GPT v2 weights from PyTorch to MLX format.

    This reuses the 1.5 conversion logic which already handles:
    - emo_conditioning_encoder
    - emo_perceiver_encoder

    Args:
        weights: PyTorch weights as numpy arrays
        config: Model configuration

    Returns:
        MLX-compatible weights
    """
    # Reuse base GPT conversion (handles shared components)
    return convert_gpt_weights(weights, config)


def convert_s2mel_weights(weights: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
    """Convert S2Mel weights from PyTorch to MLX format.

    Handles:
    - Weight normalization (weight_g, weight_v -> weight)
    - Conv1d weights transpose (PyTorch OIK -> MLX OKI)
    - Key name mapping between PyTorch and MLX

    Args:
        weights: PyTorch weights as numpy arrays

    Returns:
        MLX-compatible weights
    """
    new_weights = {}

    # Collect weight_g and weight_v pairs for weight norm
    weight_v_keys = {}
    weight_g_keys = {}
    for key in weights.keys():
        if key.endswith(".weight_v"):
            base_key = key[:-9]
            weight_v_keys[base_key] = key
        elif key.endswith(".weight_g"):
            base_key = key[:-9]
            weight_g_keys[base_key] = key

    processed_keys = set()

    for key, value in weights.items():
        new_key = key

        # Skip keys that will be processed with weight norm
        if key.endswith(".weight_g"):
            continue
        if key in processed_keys:
            continue

        # Handle weight normalization
        if key.endswith(".weight_v"):
            base_key = key[:-9]
            if base_key in weight_g_keys:
                weight_v = value
                weight_g = weights[weight_g_keys[base_key]]
                norm_dims = tuple(range(1, weight_v.ndim))
                weight_v_norm = np.sqrt(np.sum(weight_v ** 2, axis=norm_dims, keepdims=True))
                value = weight_g * (weight_v / (weight_v_norm + 1e-8))
                new_key = base_key + ".weight"
                processed_keys.add(key)
                processed_keys.add(weight_g_keys[base_key])
            else:
                new_key = base_key + ".weight"

        # Skip unused keys
        if "input_pos" in key or "freqs_cis" in key or "causal_mask" in key:
            continue

        # Key name mappings
        # gpt_layer: gpt_layer.0.* -> gpt_layer.layers.0.*
        new_key = re.sub(r"^gpt_layer\.(\d+)\.", r"gpt_layer.layers.\1.", new_key)

        # TimestepEmbedder: mlp.0 -> linear1, mlp.2 -> linear2
        new_key = re.sub(r"t_embedder\.mlp\.0\.", "t_embedder.linear1.", new_key)
        new_key = re.sub(r"t_embedder\.mlp\.2\.", "t_embedder.linear2.", new_key)
        new_key = re.sub(r"t_embedder2\.mlp\.0\.", "t_embedder2.linear1.", new_key)
        new_key = re.sub(r"t_embedder2\.mlp\.2\.", "t_embedder2.linear2.", new_key)

        # FinalLayer: adaLN_modulation.1 -> adaLN_modulation.layers.1
        new_key = re.sub(r"adaLN_modulation\.(\d+)\.", r"adaLN_modulation.layers.\1.", new_key)

        # SConv1d nested structure
        new_key = re.sub(r"\.conv\.conv\.", ".conv.", new_key)

        # Remove "models." prefix if present
        new_key = re.sub(r"^models\.", "", new_key)

        # Handle Conv1d weight transpose
        is_conv1d_weight = (
            value.ndim == 3 and
            ".weight" in new_key and
            "embedding" not in new_key
        )
        if is_conv1d_weight:
            value = value.transpose(0, 2, 1)

        new_weights[new_key] = mx.array(value)

    return new_weights


def convert_bigvgan_v2_weights(weights: Dict[str, np.ndarray]) -> Dict[str, mx.array]:
    """Convert BigVGAN v2 weights from PyTorch to MLX format.

    Handles:
    - Weight normalization (weight_g, weight_v -> weight)
    - Conv1d/ConvTranspose1d weight transposition
    - Alias-free activation filter (skip, computed at runtime)
    - Key renaming (ups.X.0 -> ups.X)

    Args:
        weights: PyTorch weights as numpy arrays

    Returns:
        MLX-compatible weights
    """
    new_weights = {}

    # Collect weight_g and weight_v pairs
    weight_v_keys = {}
    weight_g_keys = {}
    for key in weights.keys():
        if key.endswith(".weight_v"):
            base_key = key[:-9]
            weight_v_keys[base_key] = key
        elif key.endswith(".weight_g"):
            base_key = key[:-9]
            weight_g_keys[base_key] = key

    for key, value in weights.items():
        # Skip alias-free filter weights (computed at runtime)
        if "filter" in key:
            continue

        # Handle weight normalization
        if key.endswith(".weight_v"):
            base_key = key[:-9]
            if base_key in weight_g_keys:
                weight_v = value
                weight_g = weights[weight_g_keys[base_key]]
                norm_dims = tuple(range(1, weight_v.ndim))
                weight_v_norm = np.sqrt(np.sum(weight_v ** 2, axis=norm_dims, keepdims=True))
                weight = weight_g * (weight_v / (weight_v_norm + 1e-8))
                new_key = base_key + ".weight"
            else:
                weight = value
                new_key = base_key + ".weight"
        elif key.endswith(".weight_g"):
            continue
        else:
            weight = value
            new_key = key

        # Rename ups.X.0 -> ups.X
        new_key = re.sub(r"ups\.(\d+)\.0\.", r"ups.\1.", new_key)

        # Handle weight transposition for 3D weights
        if weight.ndim == 3 and "weight" in new_key:
            if new_key.startswith("ups."):
                # ConvTranspose1d: PyTorch (in, out, k) -> MLX (out, k, in)
                weight = weight.transpose(1, 2, 0)
            elif "conv" in new_key.lower():
                # Regular Conv1d: PyTorch (out, in, k) -> MLX (out, k, in)
                weight = weight.transpose(0, 2, 1)

        new_weights[new_key] = mx.array(weight)

    return new_weights


def export_vq2emb_weights(cfg) -> Dict[str, mx.array]:
    """Export vq2emb weights from Semantic Codec.

    vq2emb converts mel codes to embeddings using:
    1. Codebook embedding lookup (8192 -> 8)
    2. Linear projection via Conv1d kernel_size=1 (8 -> 1024)

    The Conv1d uses weight_norm, so we compute the actual weight as:
        weight = weight_g * weight_v / ||weight_v||

    Args:
        cfg: OmegaConf config containing semantic_codec config

    Returns:
        Dict with keys: codebook.weight, out_project.weight, out_project.bias
    """
    import torch
    import safetensors.torch
    from huggingface_hub import hf_hub_download
    from mlx_indextts.indextts.utils.maskgct_utils import build_semantic_codec

    # Build semantic codec
    semantic_codec = build_semantic_codec(cfg.semantic_codec)

    # Load weights from HuggingFace
    ckpt_path = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, ckpt_path)
    semantic_codec.eval()

    # Get the quantizer (single FVQ with num_quantizers=1)
    quantizer = semantic_codec.quantizer.quantizers[0]

    # Extract weights with weight_norm properly applied
    # weight_norm stores weight_g and weight_v, actual weight = g * v / ||v||
    weight_g = quantizer.out_project.weight_g.detach()  # (1024, 1, 1)
    weight_v = quantizer.out_project.weight_v.detach()  # (1024, 8, 1)
    norm = weight_v.norm(dim=1, keepdim=True)  # L2 norm over in_channels
    actual_weight = weight_g * weight_v / norm  # (1024, 8, 1)

    # Codebook and bias
    codebook = quantizer.codebook.weight.detach()  # (8192, 8)
    bias = quantizer.out_project.bias.detach()  # (1024,)

    # Convert to MLX
    weights = {
        "codebook.weight": mx.array(codebook.numpy()),
        "out_project.weight": mx.array(actual_weight.numpy()),
        "out_project.bias": mx.array(bias.numpy()),
    }

    return weights


def convert_model(
    model_dir: Union[str, Path],
    output_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    quantize_bits: Optional[int] = None,
) -> None:
    """Convert IndexTTS 2.0 PyTorch model to MLX format.

    Converts all three components (GPT v2, S2Mel, BigVGAN v2) to a single output directory.

    Args:
        model_dir: Directory containing PyTorch checkpoints (indexTTS-2)
        output_dir: Output directory for MLX weights
        config_path: Optional path to config.yaml
        quantize_bits: Quantization bits for GPT (4 or 8), None for fp32
    """
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if config_path is None:
        config_path = model_dir / "config.yaml"

    cfg = OmegaConf.load(str(config_path))
    config = IndexTTSConfig.from_omegaconf(cfg)

    print(f"Converting IndexTTS 2.0 model from {model_dir}")
    print(f"Output directory: {output_dir}")
    if quantize_bits:
        print(f"Quantization: {quantize_bits}-bit (GPT only)")
    else:
        print("Quantization: None (fp32)")

    # 1. Convert GPT v2 weights
    gpt_path = model_dir / "gpt.pth"
    if gpt_path.exists():
        print(f"\n[1/3] Converting GPT v2 weights from {gpt_path}...")
        gpt_numpy = load_pytorch_weights(gpt_path)
        gpt_weights = convert_gpt_v2_weights(gpt_numpy, config)

        # Apply quantization if requested
        if quantize_bits:
            print(f"  Quantizing GPT to {quantize_bits}-bit...")
            gpt_weights = _quantize_weights(gpt_weights, config, quantize_bits)

        gpt_output = output_dir / "gpt.safetensors"
        mx.save_safetensors(str(gpt_output), gpt_weights)
        print(f"  Saved {len(gpt_weights)} tensors to {gpt_output}")
    else:
        print(f"Warning: GPT checkpoint not found at {gpt_path}")

    # 2. Convert S2Mel weights
    s2mel_path = model_dir / cfg.s2mel_checkpoint
    if s2mel_path.exists():
        print(f"\n[2/3] Converting S2Mel weights from {s2mel_path}...")
        state = torch.load(str(s2mel_path), map_location="cpu")
        s2mel_params = state["net"]

        # Flatten nested structure
        s2mel_numpy = {}
        for module_name, module_params in s2mel_params.items():
            for key, value in module_params.items():
                full_key = f"{module_name}.{key}"
                if isinstance(value, torch.Tensor):
                    s2mel_numpy[full_key] = value.numpy()

        s2mel_weights = convert_s2mel_weights(s2mel_numpy)
        s2mel_output = output_dir / "s2mel.safetensors"
        mx.save_safetensors(str(s2mel_output), s2mel_weights)
        print(f"  Saved {len(s2mel_weights)} tensors to {s2mel_output}")
    else:
        print(f"Warning: S2Mel checkpoint not found at {s2mel_path}")

    # 3. Convert BigVGAN v2 weights (from HuggingFace)
    print("\n[3/3] Converting BigVGAN v2 from HuggingFace...")
    repo_id = "nvidia/bigvgan_v2_22khz_80band_256x"

    # Download and load weights
    weights_path = hf_hub_download(repo_id, "bigvgan_generator.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("generator", checkpoint)

    bigvgan_numpy = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            bigvgan_numpy[key] = value.numpy()

    bigvgan_weights = convert_bigvgan_v2_weights(bigvgan_numpy)
    bigvgan_output = output_dir / "bigvgan.safetensors"
    mx.save_safetensors(str(bigvgan_output), bigvgan_weights)
    print(f"  Saved {len(bigvgan_weights)} tensors to {bigvgan_output}")

    # 4. Export vq2emb weights from Semantic Codec
    print("\n[4/5] Exporting vq2emb weights from Semantic Codec...")
    vq2emb_weights = export_vq2emb_weights(cfg)
    vq2emb_output = output_dir / "vq2emb.safetensors"
    mx.save_safetensors(str(vq2emb_output), vq2emb_weights)
    print(f"  Saved {len(vq2emb_weights)} tensors to {vq2emb_output}")

    # 5. Copy BPE model
    bpe_path = model_dir / cfg.dataset.bpe_model
    if bpe_path.exists():
        shutil.copy(bpe_path, output_dir / "tokenizer.model")
        print(f"\nCopied BPE model to {output_dir / 'tokenizer.model'}")

    # 6. Copy config.yaml (needed for OmegaConf loading)
    config_yaml_src = config_path if config_path else model_dir / "config.yaml"
    if Path(config_yaml_src).exists():
        shutil.copy(config_yaml_src, output_dir / "config.yaml")
        print(f"Copied config.yaml to {output_dir / 'config.yaml'}")

    # 7. Save config as JSON (for reference)
    config_output = output_dir / "config.json"
    config_dict = config.to_dict()
    config_dict["version"] = 2.0
    config_dict["sample_rate"] = cfg.s2mel.preprocess_params.sr

    # Save quantization info
    if quantize_bits:
        config_dict["quantize_bits"] = quantize_bits

    # Save S2Mel specific config
    config_dict["s2mel"] = {
        "sr": cfg.s2mel.preprocess_params.sr,
        "n_fft": cfg.s2mel.preprocess_params.spect_params.n_fft,
        "hop_length": cfg.s2mel.preprocess_params.spect_params.hop_length,
        "win_length": cfg.s2mel.preprocess_params.spect_params.win_length,
        "n_mels": cfg.s2mel.preprocess_params.spect_params.n_mels,
    }

    with open(config_output, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"Saved config to {config_output}")

    # 8. Copy emotion matrices if present
    for feat_file in ["feat1.pt", "feat2.pt"]:
        feat_path = model_dir / feat_file
        if feat_path.exists():
            shutil.copy(feat_path, output_dir / feat_file)
            print(f"Copied {feat_file} to {output_dir / feat_file}")

    # 9. Copy w2v stats file
    w2v_stat_path = model_dir / cfg.w2v_stat
    if w2v_stat_path.exists():
        shutil.copy(w2v_stat_path, output_dir / w2v_stat_path.name)
        print(f"Copied {cfg.w2v_stat} to {output_dir / w2v_stat_path.name}")

    print(f"\n✓ Conversion complete! Model saved to {output_dir}")
    print("\nOutput files:")
    for f in sorted(output_dir.iterdir()):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {size_mb:.1f} MB")
