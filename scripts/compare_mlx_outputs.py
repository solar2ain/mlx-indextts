#!/usr/bin/env python3
"""Compare MLX outputs with PyTorch intermediate outputs."""

import numpy as np
import mlx.core as mx

# Paths
OUTPUT_DIR = "/tmp/alignment_test"
MODEL_DIR_MLX = "/Users/didi/Projects/mlx-indextts/models/mlx-indexTTS-1.5"


def numpy_compare(name, pt_arr, mlx_arr, atol=1e-4):
    """Compare numpy arrays and report differences."""
    if pt_arr.shape != mlx_arr.shape:
        print(f"  ❌ {name}: SHAPE MISMATCH - PT {pt_arr.shape} vs MLX {mlx_arr.shape}")
        return False

    mae = np.mean(np.abs(pt_arr - mlx_arr))
    max_diff = np.max(np.abs(pt_arr - mlx_arr))

    pt_range = f"[{pt_arr.min():.4f}, {pt_arr.max():.4f}]"
    mlx_range = f"[{mlx_arr.min():.4f}, {mlx_arr.max():.4f}]"

    status = "✅" if mae < atol else "❌"
    print(f"  {status} {name}: MAE={mae:.6f}, max_diff={max_diff:.6f}")
    print(f"      PT range: {pt_range}, MLX range: {mlx_range}")

    if mae >= atol:
        flat_pt = pt_arr.flatten()
        flat_mlx = mlx_arr.flatten()
        print(f"      PT first 5:  {flat_pt[:5]}")
        print(f"      MLX first 5: {flat_mlx[:5]}")

    return mae < atol


def main():
    print("="*60)
    print("Comparing MLX with PyTorch Outputs")
    print("="*60)

    # Load MLX model
    from mlx_indextts.generate import IndexTTS
    tts = IndexTTS.load_model(MODEL_DIR_MLX)

    # 1. Test Mel Extraction
    print("\n--- Mel Spectrogram ---")
    mel_pt = np.load(f"{OUTPUT_DIR}/mel_pt.npy")
    audio_pt = np.load(f"{OUTPUT_DIR}/audio_pt.npy")

    print(f"  PT mel shape: {mel_pt.shape}")  # (1, n_mels=100, time=664)

    audio_mlx = mx.array(audio_pt.squeeze())
    mel_mlx = tts.mel_extractor(audio_mlx)

    # MLX returns (n_mels, time) when squeeze_batch, PT is (batch, n_mels, time)
    mel_mlx_np = np.array(mel_mlx)
    print(f"  MLX mel shape (raw): {mel_mlx_np.shape}")  # Should be (n_mels=100, time=664)
    mel_mlx_np = mel_mlx_np[np.newaxis, :, :]  # -> (1, n_mels, time)
    print(f"  MLX mel shape (with batch): {mel_mlx_np.shape}")

    # Trim to same length
    min_len = min(mel_pt.shape[-1], mel_mlx_np.shape[-1])
    numpy_compare("Mel Spectrogram", mel_pt[..., :min_len], mel_mlx_np[..., :min_len], atol=0.1)

    # 2. Test Conditioning
    print("\n--- Conditioning (GPT) ---")
    cond_pt = np.load(f"{OUTPUT_DIR}/conditioning_pt.npy")

    # MLX needs mel in proper format for GPT conditioning
    mel_mlx_for_gpt = mel_mlx[None, :, :]  # (time, n_mels) -> (1, time, n_mels)
    cond_mlx = tts.gpt.get_conditioning(mel_mlx_for_gpt)

    cond_mlx_np = np.array(cond_mlx)
    numpy_compare("Conditioning", cond_pt, cond_mlx_np, atol=0.01)

    # 3. Test ECAPA-TDNN (Speaker Encoder)
    print("\n--- ECAPA-TDNN Speaker Encoder ---")
    spk_emb_pt = np.load(f"{OUTPUT_DIR}/speaker_emb_pt.npy")

    # BigVGAN speaker encoder expects mel_ref in NCL format
    # PT saved mel in NCL after transpose from original
    ecapa_input_pt = np.load(f"{OUTPUT_DIR}/ecapa_input_ncl.npy")
    print(f"  PT ECAPA input shape: {ecapa_input_pt.shape}")  # Should be (1, 100, 664)

    # MLX: mel_mlx_np is already (1, n_mels=100, time=664) - same as PT
    mel_ncl_trimmed = mel_mlx_np[:, :, :ecapa_input_pt.shape[-1]]
    print(f"  MLX ECAPA input shape: {mel_ncl_trimmed.shape}")

    spk_emb_mlx = tts.bigvgan.speaker_encoder(mx.array(mel_ncl_trimmed))
    spk_emb_mlx_np = np.array(spk_emb_mlx)

    print(f"  Speaker emb shapes: PT {spk_emb_pt.shape}, MLX {spk_emb_mlx_np.shape}")
    numpy_compare("Speaker Embedding", spk_emb_pt, spk_emb_mlx_np, atol=0.01)

    # 4. Test ECAPA-TDNN intermediate layers
    print("\n--- ECAPA-TDNN Intermediate Layers ---")

    ecapa = tts.bigvgan.speaker_encoder
    x_mlx = mx.array(ecapa_input_pt)  # Use PT input for exact comparison

    # Test block 0
    block0_pt = np.load(f"{OUTPUT_DIR}/ecapa_block0_out.npy")
    block0_mlx = ecapa.blocks[0](x_mlx)
    block0_mlx_np = np.array(block0_mlx)
    numpy_compare("Block 0 (TDNNBlock)", block0_pt, block0_mlx_np, atol=0.001)

    # Test remaining blocks
    x_block = mx.array(block0_pt)  # Use PT output to isolate each block
    for i in range(1, 4):
        block_pt = np.load(f"{OUTPUT_DIR}/ecapa_block{i}_out.npy")
        x_block = ecapa.blocks[i](x_block)
        block_mlx_np = np.array(x_block)
        numpy_compare(f"Block {i} (SERes2NetBlock)", block_pt, block_mlx_np, atol=0.001)
        x_block = mx.array(block_pt)  # Reset to PT output

    # Test MFA
    mfa_in_pt = np.load(f"{OUTPUT_DIR}/ecapa_mfa_input.npy")
    mfa_out_pt = np.load(f"{OUTPUT_DIR}/ecapa_mfa_out.npy")
    mfa_out_mlx = ecapa.mfa(mx.array(mfa_in_pt))
    mfa_out_mlx_np = np.array(mfa_out_mlx)
    numpy_compare("MFA", mfa_out_pt, mfa_out_mlx_np, atol=0.001)

    # Test ASP
    asp_out_pt = np.load(f"{OUTPUT_DIR}/ecapa_asp_out.npy")
    asp_out_mlx = ecapa.asp(mx.array(mfa_out_pt))
    asp_out_mlx_np = np.array(asp_out_mlx)
    numpy_compare("ASP", asp_out_pt, asp_out_mlx_np, atol=0.001)

    # Test ASP BN
    asp_bn_pt = np.load(f"{OUTPUT_DIR}/ecapa_asp_bn_out.npy")
    asp_bn_mlx = ecapa.asp_bn(mx.array(asp_out_pt).transpose(0, 2, 1))  # MLX BatchNorm expects NLC
    asp_bn_mlx_np = np.array(asp_bn_mlx.transpose(0, 2, 1))  # Back to NCL
    numpy_compare("ASP BN", asp_bn_pt, asp_bn_mlx_np, atol=0.001)

    # Test FC
    fc_pt = np.load(f"{OUTPUT_DIR}/ecapa_fc_out.npy")
    fc_mlx = ecapa.fc(mx.array(asp_bn_pt).transpose(0, 2, 1))  # MLX Conv1d expects NLC
    fc_mlx_np = np.array(fc_mlx.transpose(0, 2, 1))  # Back to NCL
    numpy_compare("FC", fc_pt, fc_mlx_np, atol=0.001)

    # 5. Test Text Tokenization
    print("\n--- Text Tokenization ---")
    tokens_pt = np.load(f"{OUTPUT_DIR}/text_tokens.npy")
    text = "你好"
    tokens_mlx = tts.tokenizer.encode(text)
    print(f"  PT tokens: {tokens_pt.tolist()}")
    print(f"  MLX tokens: {tokens_mlx}")
    if list(tokens_pt) == tokens_mlx:
        print("  ✅ Tokens match!")
    else:
        print("  ❌ Tokens mismatch!")

    # 6. Test GPT Latent (with same codes)
    print("\n--- GPT Latent ---")
    mel_codes = np.load(f"{OUTPUT_DIR}/mel_codes.npy")
    gpt_latent_pt = np.load(f"{OUTPUT_DIR}/gpt_latent.npy")

    codes_mlx = mx.array(mel_codes[:, :50])  # Match PyTorch: codes[:, :50]
    text_tokens_mlx = mx.array(tokens_pt, dtype=mx.int32)[None, :]

    # Get conditioning
    cond_mlx_for_latent = tts.gpt.get_conditioning(mel_mlx_for_gpt)

    latent_mlx = tts.gpt.forward_latent(cond_mlx_for_latent, text_tokens_mlx, codes_mlx)
    latent_mlx_np = np.array(latent_mlx)

    print(f"  PT latent shape: {gpt_latent_pt.shape}, MLX latent shape: {latent_mlx_np.shape}")
    numpy_compare("GPT Latent", gpt_latent_pt, latent_mlx_np, atol=0.01)

    # 7. Test BigVGAN output
    print("\n--- BigVGAN Output ---")
    wav_pt = np.load(f"{OUTPUT_DIR}/wav_pt.npy")

    # Use PT latent for exact comparison
    latent_for_bigvgan = mx.array(gpt_latent_pt)
    mel_ref_for_bigvgan = mel_mlx_for_gpt  # Already (1, time, n_mels)

    # BigVGAN expects mel_ref in NCL format
    mel_ref_ncl = mx.array(mel_pt[:, :, :664])

    wav_mlx = tts.bigvgan(latent_for_bigvgan, mel_ref_ncl)
    wav_mlx_np = np.array(wav_mlx)

    print(f"  PT wav shape: {wav_pt.shape}, MLX wav shape: {wav_mlx_np.shape}")

    # Trim to same length
    min_wav_len = min(wav_pt.shape[-1], wav_mlx_np.shape[-1])
    numpy_compare("BigVGAN Output", wav_pt[..., :min_wav_len], wav_mlx_np[..., :min_wav_len], atol=0.1)

    print("\n" + "="*60)
    print("Alignment Test Complete")
    print("="*60)


if __name__ == "__main__":
    main()
