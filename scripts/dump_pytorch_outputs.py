#!/usr/bin/env python3
"""Dump PyTorch intermediate outputs for alignment testing."""

import sys
sys.path.insert(0, "/Users/didi/Projects/index-tts")

import os
import numpy as np
import torch
import soundfile as sf
import torchaudio
import pickle

# Paths
REF_AUDIO_PATH = "/Users/didi/.openclaw-eridu/workspace-elina/voices/elina.wav"
MODEL_DIR = "/Users/didi/Projects/index-tts/indexTTS-1.5"
OUTPUT_DIR = "/tmp/alignment_test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("="*60)
    print("Dumping PyTorch Intermediate Outputs")
    print("="*60)

    # Load audio
    audio_np, sr = sf.read(REF_AUDIO_PATH)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    audio_pt = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0)
    audio_pt = torchaudio.transforms.Resample(sr, 24000)(audio_pt)

    print(f"Audio shape: {audio_pt.shape}")

    # 1. Mel extraction
    from indextts.utils.feature_extractors import MelSpectrogramFeatures
    mel_extractor = MelSpectrogramFeatures()
    mel_pt = mel_extractor(audio_pt)
    print(f"Mel shape: {mel_pt.shape}")

    np.save(f"{OUTPUT_DIR}/mel_pt.npy", mel_pt.numpy())
    np.save(f"{OUTPUT_DIR}/audio_pt.npy", audio_pt.numpy())

    # 2. Load GPT model and get conditioning
    from omegaconf import OmegaConf
    from indextts.gpt.model import UnifiedVoice
    from indextts.utils.checkpoint import load_checkpoint

    cfg = OmegaConf.load(f"{MODEL_DIR}/config.yaml")
    gpt = UnifiedVoice(**cfg.gpt)
    load_checkpoint(gpt, f"{MODEL_DIR}/gpt.pth")
    gpt.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    gpt = gpt.to(device)
    mel_pt = mel_pt.to(device)

    cond_mel_lengths = torch.tensor([mel_pt.shape[-1]], device=device)

    with torch.no_grad():
        # Get conditioning
        cond_pt = gpt.get_conditioning(mel_pt, cond_mel_lengths=cond_mel_lengths)
        print(f"Conditioning shape: {cond_pt.shape}")

        # Also save conformer output
        # Conformer expects (batch, time, n_mels) - NLC format
        mel_nlc = mel_pt.transpose(1, 2)  # NCL -> NLC

        # Save subsampling output
        masks = ~torch.zeros((1, mel_nlc.shape[1]), dtype=torch.bool, device=device).unsqueeze(1)
        embed_out, pos_emb, new_masks = gpt.conditioning_encoder.embed(mel_nlc, masks)
        print(f"Embed output shape: {embed_out.shape}")
        np.save(f"{OUTPUT_DIR}/embed_out_pt.npy", embed_out.cpu().numpy())
        np.save(f"{OUTPUT_DIR}/pos_emb_pt.npy", pos_emb.cpu().numpy())

        conformer_out, mask = gpt.conditioning_encoder(mel_nlc, cond_mel_lengths)
        print(f"Conformer output shape: {conformer_out.shape}")
        np.save(f"{OUTPUT_DIR}/conformer_out_pt.npy", conformer_out.cpu().numpy())

        # Perceiver output
        perceiver_out = gpt.perceiver_encoder(conformer_out)
        print(f"Perceiver output shape: {perceiver_out.shape}")
        np.save(f"{OUTPUT_DIR}/perceiver_out_pt.npy", perceiver_out.cpu().numpy())

    np.save(f"{OUTPUT_DIR}/conditioning_pt.npy", cond_pt.cpu().numpy())

    # 3. Load BigVGAN and test speaker encoder
    from indextts.BigVGAN.models import BigVGAN as Generator

    bigvgan = Generator(cfg.bigvgan, use_cuda_kernel=False)
    vocoder_dict = torch.load(f"{MODEL_DIR}/bigvgan_generator.pth", map_location="cpu")
    bigvgan.load_state_dict(vocoder_dict["generator"])
    bigvgan = bigvgan.to(device)
    bigvgan.remove_weight_norm()
    bigvgan.eval()

    # mel_ref should be (batch, time, n_mels) for speaker_encoder
    mel_ref = mel_pt.transpose(1, 2)  # (1, n_mels, time) -> (1, time, n_mels)
    print(f"Mel ref shape for speaker encoder: {mel_ref.shape}")

    with torch.no_grad():
        spk_emb_pt = bigvgan.speaker_encoder(mel_ref)
        print(f"Speaker embedding shape: {spk_emb_pt.shape}")

    np.save(f"{OUTPUT_DIR}/speaker_emb_pt.npy", spk_emb_pt.cpu().numpy())

    # 4. Test ECAPA-TDNN intermediate outputs
    print("\n--- ECAPA-TDNN Intermediate Outputs ---")

    ecapa = bigvgan.speaker_encoder
    x = mel_ref.to(device)

    # Original ECAPA expects NLC (batch, time, n_mels), internal uses NCL
    # Let's trace through the forward pass

    # Step 1: Input processing
    # Original does: x = x.transpose(1, 2) to get NCL
    x_ncl = x.transpose(1, 2)  # Now NCL: (batch, n_mels, time)
    print(f"After transpose to NCL: {x_ncl.shape}")

    np.save(f"{OUTPUT_DIR}/ecapa_input_ncl.npy", x_ncl.cpu().numpy())

    # Step 2: First block (TDNNBlock)
    with torch.no_grad():
        x_after_block0 = ecapa.blocks[0](x_ncl)
        print(f"After block 0: {x_after_block0.shape}")
        np.save(f"{OUTPUT_DIR}/ecapa_block0_out.npy", x_after_block0.cpu().numpy())

        # Collect all block outputs
        xl = [x_after_block0]
        x_block = x_after_block0
        for i, block in enumerate(ecapa.blocks[1:], 1):
            x_block = block(x_block)
            xl.append(x_block)
            print(f"After block {i}: {x_block.shape}")
            np.save(f"{OUTPUT_DIR}/ecapa_block{i}_out.npy", x_block.cpu().numpy())

        # MFA (Multi-layer Feature Aggregation)
        x_mfa_in = torch.cat(xl[1:], dim=1)
        print(f"MFA input (concat of blocks 1-3): {x_mfa_in.shape}")
        np.save(f"{OUTPUT_DIR}/ecapa_mfa_input.npy", x_mfa_in.cpu().numpy())

        x_mfa = ecapa.mfa(x_mfa_in)
        print(f"After MFA: {x_mfa.shape}")
        np.save(f"{OUTPUT_DIR}/ecapa_mfa_out.npy", x_mfa.cpu().numpy())

        # ASP (Attentive Statistics Pooling)
        x_asp = ecapa.asp(x_mfa)
        print(f"After ASP: {x_asp.shape}")
        np.save(f"{OUTPUT_DIR}/ecapa_asp_out.npy", x_asp.cpu().numpy())

        # ASP BatchNorm
        x_asp_bn = ecapa.asp_bn(x_asp)
        print(f"After ASP BN: {x_asp_bn.shape}")
        np.save(f"{OUTPUT_DIR}/ecapa_asp_bn_out.npy", x_asp_bn.cpu().numpy())

        # FC
        x_fc = ecapa.fc(x_asp_bn)
        print(f"After FC: {x_fc.shape}")
        np.save(f"{OUTPUT_DIR}/ecapa_fc_out.npy", x_fc.cpu().numpy())

    # 5. Test text tokenization
    from indextts.utils.front import TextNormalizer, TextTokenizer
    normalizer = TextNormalizer()
    normalizer.load()
    tokenizer = TextTokenizer(f"{MODEL_DIR}/bpe.model", normalizer)

    text = "你好"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\nText: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")

    np.save(f"{OUTPUT_DIR}/text_tokens.npy", np.array(ids))

    # 6. Generate mel codes with fixed seed
    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)

    text_tokens = torch.tensor(ids, dtype=torch.int32, device=device).unsqueeze(0)
    cond_mel_lengths = torch.tensor([mel_pt.shape[-1]], device=device)

    torch.manual_seed(42)
    with torch.no_grad():
        codes = gpt.inference_speech(
            mel_pt,
            text_tokens,
            cond_mel_lengths=cond_mel_lengths,
            do_sample=True,
            top_p=0.8,
            top_k=30,
            temperature=1.0,
            max_generate_length=100,
        )
    print(f"\nGenerated codes shape: {codes.shape}")
    print(f"Codes[:20]: {codes[0, :20].tolist()}")

    np.save(f"{OUTPUT_DIR}/mel_codes.npy", codes.cpu().numpy())

    # 7. Get latents
    codes_short = codes[:, :50]
    code_lens = torch.tensor([codes_short.shape[-1]], device=device, dtype=codes_short.dtype)

    with torch.no_grad():
        latent = gpt(
            mel_pt,
            text_tokens,
            torch.tensor([text_tokens.shape[-1]], device=device),
            codes_short,
            code_lens * gpt.mel_length_compression,
            cond_mel_lengths=cond_mel_lengths,
            return_latent=True,
            clip_inputs=False
        )
    print(f"Latent shape: {latent.shape}")
    np.save(f"{OUTPUT_DIR}/gpt_latent.npy", latent.cpu().numpy())

    # 8. BigVGAN output
    with torch.no_grad():
        wav, _ = bigvgan(latent, mel_ref)
    print(f"Wav shape: {wav.shape}")
    np.save(f"{OUTPUT_DIR}/wav_pt.npy", wav.cpu().numpy())

    print(f"\n✅ All outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
