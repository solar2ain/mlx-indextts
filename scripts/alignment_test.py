#!/usr/bin/env python3
"""
Systematic alignment test between PyTorch and MLX implementations.

This script records intermediate outputs from both implementations
and compares them layer by layer to identify precision mismatches.
"""

import sys
import os

# Add paths
sys.path.insert(0, "/Users/didi/Projects/index-tts")
sys.path.insert(0, "/Users/didi/Projects/mlx-indextts/src")

import numpy as np
import torch
import mlx.core as mx

# Shared test parameters
REF_AUDIO_PATH = "/Users/didi/.openclaw-eridu/workspace-elina/voices/elina.wav"
TEXT = "你好"
MODEL_DIR_PYTORCH = "/Users/didi/Projects/index-tts/indexTTS-1.5"
MODEL_DIR_MLX = "/Users/didi/Projects/mlx-indextts/models/mlx-indexTTS-1.5"


def numpy_compare(name: str, pt_arr: np.ndarray, mlx_arr: np.ndarray, atol=1e-4):
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
        # Show sample values
        flat_pt = pt_arr.flatten()
        flat_mlx = mlx_arr.flatten()
        print(f"      PT first 5:  {flat_pt[:5]}")
        print(f"      MLX first 5: {flat_mlx[:5]}")

    return mae < atol


def load_audio_pt(path, target_sr=24000):
    """Load audio for PyTorch using soundfile."""
    import soundfile as sf
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    if sr != target_sr:
        import torchaudio
        audio = torchaudio.transforms.Resample(sr, target_sr)(audio)
    return audio, target_sr


def test_mel_extraction():
    """Test mel spectrogram extraction alignment."""
    print("\n" + "="*60)
    print("TEST: Mel Spectrogram Extraction")
    print("="*60)

    import soundfile as sf

    # Load audio
    audio_pt, sr = load_audio_pt(REF_AUDIO_PATH, 24000)

    # PyTorch mel extraction
    from indextts.utils.feature_extractors import MelSpectrogramFeatures
    mel_extractor_pt = MelSpectrogramFeatures()
    mel_pt = mel_extractor_pt(audio_pt)

    # MLX mel extraction
    from mlx_indextts.mel import MelSpectrogramExtractor
    audio_mlx = mx.array(audio_pt.numpy().squeeze())
    mel_extractor_mlx = MelSpectrogramExtractor(
        n_fft=1024, hop_length=256, win_length=1024,
        n_mels=100, sample_rate=24000, f_min=0, f_max=12000
    )
    mel_mlx = mel_extractor_mlx(audio_mlx)

    print(f"  PT mel shape: {mel_pt.shape}")
    print(f"  MLX mel shape: {mel_mlx.shape}")

    # Convert for comparison
    mel_pt_np = mel_pt.numpy()
    mel_mlx_np = np.array(mel_mlx)

    # MLX returns (time, n_mels), PT returns (batch, n_mels, time)
    if mel_mlx_np.ndim == 2:
        mel_mlx_np = mel_mlx_np.T[np.newaxis, :, :]  # (time, mels) -> (1, mels, time)

    numpy_compare("Mel Spectrogram", mel_pt_np, mel_mlx_np, atol=0.1)

    return mel_pt, mel_mlx


def test_conditioning_encoder():
    """Test conditioning encoder alignment."""
    print("\n" + "="*60)
    print("TEST: Conditioning Encoder")
    print("="*60)

    mel_pt, mel_mlx = test_mel_extraction()

    # Load PyTorch model
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    cfg = OmegaConf.load(f"{MODEL_DIR_PYTORCH}/config.yaml")
    gpt_pt = UnifiedVoice(**cfg.gpt)
    gpt_path = f"{MODEL_DIR_PYTORCH}/gpt.pth"
    load_checkpoint(gpt_pt, gpt_path)
    gpt_pt.eval()

    # Load MLX model
    from mlx_indextts.generate import IndexTTS
    tts_mlx = IndexTTS.load_model(MODEL_DIR_MLX)

    # Get conditioning from PyTorch
    with torch.no_grad():
        cond_pt = gpt_pt.get_conditioning(mel_pt.cuda() if torch.cuda.is_available() else mel_pt)

    # Get conditioning from MLX
    if mel_mlx.ndim == 2:
        mel_mlx = mel_mlx[None, :, :]  # Add batch dim
    # MLX ECAPA expects input in proper format
    cond_mlx = tts_mlx.gpt.get_conditioning(mel_mlx)

    print(f"  PT conditioning shape: {cond_pt.shape}")
    print(f"  MLX conditioning shape: {cond_mlx.shape}")

    cond_pt_np = cond_pt.cpu().numpy()
    cond_mlx_np = np.array(cond_mlx)

    numpy_compare("Conditioning Latents", cond_pt_np, cond_mlx_np, atol=0.01)

    return gpt_pt, tts_mlx, mel_pt, mel_mlx, cond_pt, cond_mlx


def test_text_tokenization():
    """Test text tokenization alignment."""
    print("\n" + "="*60)
    print("TEST: Text Tokenization")
    print("="*60)

    # PyTorch tokenizer
    from indextts.utils.front import TextNormalizer, TextTokenizer
    normalizer_pt = TextNormalizer()
    normalizer_pt.load()
    tokenizer_pt = TextTokenizer(f"{MODEL_DIR_PYTORCH}/bpe.model", normalizer_pt)

    tokens_pt = tokenizer_pt.tokenize(TEXT)
    ids_pt = tokenizer_pt.convert_tokens_to_ids(tokens_pt)

    # MLX tokenizer
    from mlx_indextts.tokenizer import TextTokenizer as MLXTextTokenizer
    from mlx_indextts.normalize import TextNormalizer as MLXTextNormalizer
    normalizer_mlx = MLXTextNormalizer()
    normalizer_mlx.load()
    tokenizer_mlx = MLXTextTokenizer(f"{MODEL_DIR_MLX}/tokenizer.model", normalizer_mlx)

    ids_mlx = tokenizer_mlx.encode(TEXT)

    print(f"  PT tokens: {tokens_pt}")
    print(f"  PT ids: {ids_pt}")
    print(f"  MLX ids: {ids_mlx}")

    if ids_pt == ids_mlx:
        print("  ✅ Token IDs match!")
    else:
        print("  ❌ Token IDs mismatch!")

    return ids_pt, ids_mlx


def test_gpt_generation():
    """Test GPT generation alignment."""
    print("\n" + "="*60)
    print("TEST: GPT Generation (Using Fixed Seed)")
    print("="*60)

    # First get all prerequisites
    gpt_pt, tts_mlx, mel_pt, mel_mlx, cond_pt, cond_mlx = test_conditioning_encoder()
    ids_pt, ids_mlx = test_text_tokenization()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt_pt = gpt_pt.to(device)

    # Setup for generation
    gpt_pt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

    text_tokens_pt = torch.tensor(ids_pt, dtype=torch.int32, device=device).unsqueeze(0)
    cond_mel_lengths = torch.tensor([mel_pt.shape[-1]], device=device)

    # Generate codes with PyTorch (use deterministic settings)
    torch.manual_seed(42)
    with torch.no_grad():
        codes_pt = gpt_pt.inference_speech(
            mel_pt.to(device),
            text_tokens_pt,
            cond_mel_lengths=cond_mel_lengths,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=1.0,
            max_generate_length=100,  # Short for testing
        )

    print(f"  PT generated codes shape: {codes_pt.shape}")
    print(f"  PT codes[:20]: {codes_pt[0, :20].tolist()}")

    # Note: MLX generation uses different random state, so codes will differ
    # The key is to test the latent computation given the SAME codes

    return gpt_pt, tts_mlx, mel_pt, mel_mlx, cond_pt, cond_mlx, codes_pt


def test_latent_extraction():
    """Test latent extraction with same codes."""
    print("\n" + "="*60)
    print("TEST: Latent Extraction (Same Codes)")
    print("="*60)

    gpt_pt, tts_mlx, mel_pt, mel_mlx, cond_pt, cond_mlx, codes_pt = test_gpt_generation()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids_pt, _ = test_text_tokenization()

    text_tokens_pt = torch.tensor(ids_pt, dtype=torch.int32, device=device).unsqueeze(0)

    # Use the same codes for both
    codes = codes_pt[:, :50]  # Take first 50 codes for testing
    code_lens = torch.tensor([codes.shape[-1]], device=device, dtype=codes.dtype)

    # Get latent from PyTorch
    with torch.no_grad():
        latent_pt = gpt_pt(
            mel_pt.to(device),
            text_tokens_pt,
            torch.tensor([text_tokens_pt.shape[-1]], device=device),
            codes,
            code_lens * gpt_pt.mel_length_compression,
            cond_mel_lengths=torch.tensor([mel_pt.shape[-1]], device=device),
            return_latent=True,
            clip_inputs=False
        )

    print(f"  PT latent shape: {latent_pt.shape}")

    # Get latent from MLX using same codes
    codes_mlx = mx.array(codes.cpu().numpy())
    text_tokens_mlx = mx.array(ids_pt, dtype=mx.int32)[None, :]

    latent_mlx = tts_mlx.gpt.forward_latent(cond_mlx, text_tokens_mlx, codes_mlx)

    print(f"  MLX latent shape: {latent_mlx.shape}")

    latent_pt_np = latent_pt.cpu().numpy()
    latent_mlx_np = np.array(latent_mlx)

    numpy_compare("GPT Latents", latent_pt_np, latent_mlx_np, atol=0.01)

    return gpt_pt, tts_mlx, mel_pt, mel_mlx, latent_pt, latent_mlx


def test_bigvgan():
    """Test BigVGAN vocoder alignment."""
    print("\n" + "="*60)
    print("TEST: BigVGAN Vocoder")
    print("="*60)

    gpt_pt, tts_mlx, mel_pt, mel_mlx, latent_pt, latent_mlx = test_latent_extraction()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load PyTorch BigVGAN
    from omegaconf import OmegaConf
    from indextts.BigVGAN.models import BigVGAN as Generator

    cfg = OmegaConf.load(f"{MODEL_DIR_PYTORCH}/config.yaml")
    bigvgan_pt = Generator(cfg.bigvgan, use_cuda_kernel=False)
    vocoder_dict = torch.load(f"{MODEL_DIR_PYTORCH}/bigvgan_generator.pth", map_location="cpu")
    bigvgan_pt.load_state_dict(vocoder_dict["generator"])
    bigvgan_pt = bigvgan_pt.to(device)
    bigvgan_pt.remove_weight_norm()
    bigvgan_pt.eval()

    # Test BigVGAN with same inputs
    print("\n  Testing BigVGAN forward pass...")

    # PyTorch BigVGAN expects: (latent, mel_ref.transpose(1, 2))
    # mel_ref should be (batch, time, n_mels) after transpose
    mel_ref_pt = mel_pt.to(device).transpose(1, 2)

    with torch.no_grad():
        wav_pt, _ = bigvgan_pt(latent_pt.to(device), mel_ref_pt)

    print(f"  PT wav shape: {wav_pt.shape}")

    # MLX BigVGAN
    # mel_mlx should be in proper format
    if mel_mlx.ndim == 2:
        mel_mlx_input = mel_mlx[None, :, :]
    else:
        mel_mlx_input = mel_mlx

    wav_mlx = tts_mlx.bigvgan(latent_mlx, mel_mlx_input)

    print(f"  MLX wav shape: {wav_mlx.shape}")

    wav_pt_np = wav_pt.cpu().numpy()
    wav_mlx_np = np.array(wav_mlx)

    # Align shapes for comparison
    if wav_pt_np.shape != wav_mlx_np.shape:
        min_len = min(wav_pt_np.shape[-1], wav_mlx_np.shape[-1])
        wav_pt_np = wav_pt_np[..., :min_len]
        wav_mlx_np = wav_mlx_np[..., :min_len]

    numpy_compare("BigVGAN Output", wav_pt_np, wav_mlx_np, atol=0.1)


def test_bigvgan_speaker_encoder_detailed():
    """Detailed test of BigVGAN's speaker encoder (ECAPA-TDNN)."""
    print("\n" + "="*60)
    print("TEST: BigVGAN Speaker Encoder (ECAPA-TDNN) - Detailed")
    print("="*60)

    # Load audio
    audio_pt, sr = load_audio_pt(REF_AUDIO_PATH, 24000)

    # Get mel
    from indextts.utils.feature_extractors import MelSpectrogramFeatures
    mel_extractor_pt = MelSpectrogramFeatures()
    mel_pt = mel_extractor_pt(audio_pt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load PyTorch BigVGAN
    from omegaconf import OmegaConf
    from indextts.BigVGAN.models import BigVGAN as Generator

    cfg = OmegaConf.load(f"{MODEL_DIR_PYTORCH}/config.yaml")
    bigvgan_pt = Generator(cfg.bigvgan, use_cuda_kernel=False)
    vocoder_dict = torch.load(f"{MODEL_DIR_PYTORCH}/bigvgan_generator.pth", map_location="cpu")
    bigvgan_pt.load_state_dict(vocoder_dict["generator"])
    bigvgan_pt = bigvgan_pt.to(device)
    bigvgan_pt.remove_weight_norm()
    bigvgan_pt.eval()

    # Load MLX model
    from mlx_indextts.generate import IndexTTS
    tts_mlx = IndexTTS.load_model(MODEL_DIR_MLX)

    # Test speaker encoder
    # PyTorch: mel_ref should be (batch, time, n_mels) - NLC format
    mel_ref_pt = mel_pt.to(device).transpose(1, 2)  # (1, n_mels, time) -> (1, time, n_mels)

    print(f"  Input mel shape: {mel_ref_pt.shape} (batch, time, n_mels)")

    with torch.no_grad():
        spk_emb_pt = bigvgan_pt.speaker_encoder(mel_ref_pt)

    print(f"  PT speaker embedding shape: {spk_emb_pt.shape}")

    # MLX: BigVGAN expects mel_ref in NCL format (batch, n_mels, time)
    from mlx_indextts.mel import MelSpectrogramExtractor
    audio_mlx = mx.array(audio_pt.numpy().squeeze())
    mel_extractor_mlx = MelSpectrogramExtractor(
        n_fft=1024, hop_length=256, win_length=1024,
        n_mels=100, sample_rate=24000, f_min=0, f_max=12000
    )
    mel_mlx = mel_extractor_mlx(audio_mlx)

    # mel_mlx is (time, n_mels), convert to (batch, n_mels, time)
    mel_mlx_ncl = mel_mlx.T[None, :, :]

    print(f"  MLX mel input shape: {mel_mlx_ncl.shape} (batch, n_mels, time)")

    spk_emb_mlx = tts_mlx.bigvgan.speaker_encoder(mel_mlx_ncl)

    print(f"  MLX speaker embedding shape: {spk_emb_mlx.shape}")

    spk_pt_np = spk_emb_pt.cpu().numpy()
    spk_mlx_np = np.array(spk_emb_mlx)

    # Reshape if needed
    if spk_pt_np.shape != spk_mlx_np.shape:
        print(f"  Shape mismatch: PT {spk_pt_np.shape} vs MLX {spk_mlx_np.shape}")
        # Try to align
        if spk_pt_np.ndim == 3 and spk_mlx_np.ndim == 3:
            # PT might be (B, 1, D), MLX might be (B, 1, D) or (B, D, 1)
            pass

    numpy_compare("Speaker Embedding", spk_pt_np, spk_mlx_np, atol=0.01)


def main():
    """Run all alignment tests."""
    print("="*60)
    print("MLX IndexTTS Alignment Tests")
    print("="*60)
    print(f"Reference audio: {REF_AUDIO_PATH}")
    print(f"Test text: {TEXT}")
    print(f"PyTorch model: {MODEL_DIR_PYTORCH}")
    print(f"MLX model: {MODEL_DIR_MLX}")

    try:
        # Run tests in order of dependencies
        test_bigvgan_speaker_encoder_detailed()
        test_bigvgan()
    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
