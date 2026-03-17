"""Debug vq2emb MLX implementation vs PyTorch."""

import sys
sys.path.insert(0, "/Users/didi/Projects/index-tts")

import torch
import numpy as np
import mlx.core as mx
import safetensors
from safetensors import safe_open
from indextts.utils.maskgct_utils import build_semantic_codec, JsonHParams
from huggingface_hub import hf_hub_download

# Build and load PyTorch semantic codec
print("Loading PyTorch Semantic Codec...")
cfg = JsonHParams(
    codebook_size=8192,
    hidden_size=1024,
    codebook_dim=8,
    vocos_dim=384,
    vocos_intermediate_dim=2048,
    vocos_num_layers=12,
    num_quantizers=1,
    downsample_scale=1,
)
semantic_codec = build_semantic_codec(cfg)
ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
safetensors.torch.load_model(semantic_codec, ckpt)
semantic_codec.eval()

quantizer = semantic_codec.quantizer.quantizers[0]

# Create test input - simulate semantic codes
torch.manual_seed(42)
# vq2emb input is (batch, length), output is (batch, channels, length)
test_codes = torch.randint(0, 8192, (1, 100))  # 100 frames
print(f"Test input shape: {test_codes.shape}")

# PyTorch forward
with torch.no_grad():
    # decode_code: embedding lookup → transpose
    emb = quantizer.decode_code(test_codes)  # (1, 8, 100)
    print(f"After decode_code (emb): {emb.shape}")
    print(f"emb[0, :4, 0]: {emb[0, :4, 0]}")

    # out_project: WNConv1d (1024, 8, 1)
    pytorch_out = quantizer.out_project(emb)  # (1, 1024, 100)
    print(f"\nPyTorch output shape: {pytorch_out.shape}")
    print(f"pytorch_out[0, :4, 0]: {pytorch_out[0, :4, 0]}")

    # Full vq2emb
    pytorch_full = quantizer.vq2emb(test_codes)
    print(f"pytorch_full[0, :4, 0]: {pytorch_full[0, :4, 0]}")

# Now MLX implementation
print("\n" + "="*50)
print("MLX Implementation")
print("="*50)

# Load MLX weights
mlx_weights = {}
with safe_open("models/mlx-indexTTS-2.0/vq2emb.safetensors", framework="numpy") as f:
    for key in f.keys():
        mlx_weights[key] = mx.array(f.get_tensor(key))
        print(f"Loaded {key}: {mlx_weights[key].shape}")

# MLX forward
codes_mx = mx.array(test_codes.numpy())

# Step 1: decode_code equivalent
# embedding lookup + transpose
codebook = mlx_weights["codebook.weight"]  # (8192, 8)
emb_mx = codebook[codes_mx]  # (1, 100, 8)
emb_mx = emb_mx.transpose(0, 2, 1)  # (1, 8, 100) to match PyTorch
mx.eval(emb_mx)
print(f"\nMLX emb shape: {emb_mx.shape}")
print(f"emb_mx[0, :4, 0]: {np.array(emb_mx[0, :4, 0])}")

# Compare embeddings
emb_diff = np.abs(np.array(emb_mx) - emb.numpy()).max()
print(f"Embedding max diff: {emb_diff}")

# Step 2: out_project - Conv1d with kernel_size=1
# PyTorch Conv1d: input (N, C_in, L), weight (C_out, C_in, K), output (N, C_out, L)
# For kernel_size=1: output[n, c_out, l] = bias[c_out] + sum_c_in(input[n, c_in, l] * weight[c_out, c_in, 0])

out_weight_mx = mlx_weights["out_project.weight"]  # (1024, 8, 1)
out_bias_mx = mlx_weights["out_project.bias"]  # (1024,)

print(f"\nout_project.weight shape: {out_weight_mx.shape}")
print(f"out_project.weight[0, :4, 0]: {np.array(out_weight_mx[0, :4, 0])}")

# Method: kernel_size=1 Conv1d is equivalent to linear transform
# emb: (1, 8, 100) -> transpose -> (1, 100, 8)
# weight: (1024, 8, 1) -> squeeze -> (1024, 8)
# matmul: (1, 100, 8) @ (8, 1024) = (1, 100, 1024)
# add bias: (1, 100, 1024) + (1024,) = (1, 100, 1024)
# transpose: (1, 1024, 100)

emb_transposed = emb_mx.transpose(0, 2, 1)  # (1, 100, 8)
weight_2d = out_weight_mx.squeeze(-1)  # (1024, 8)
mlx_out = emb_transposed @ weight_2d.T + out_bias_mx  # (1, 100, 1024)
mlx_out = mlx_out.transpose(0, 2, 1)  # (1, 1024, 100)
mx.eval(mlx_out)

print(f"\nMLX output shape: {mlx_out.shape}")
print(f"mlx_out[0, :4, 0]: {np.array(mlx_out[0, :4, 0])}")

# Compare
diff = np.abs(np.array(mlx_out) - pytorch_out.numpy())
print(f"\nMax diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")

if diff.max() < 1e-5:
    print("\n✅ MLX vq2emb matches PyTorch!")
else:
    print("\n❌ Mismatch detected")
    print(f"PyTorch out_project.weight[0, :4, 0]: {quantizer.out_project.weight[0, :4, 0]}")
