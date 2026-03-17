#!/usr/bin/env python3
"""Dump PyTorch intermediate outputs for IndexTTS 2.0 alignment testing.

This script runs the original PyTorch IndexTTS 2.0 and saves intermediate outputs
at each stage for comparison with MLX implementation.

Outputs are saved to:
- test_data/gpt_v2/        - GPT v2 conditioning outputs
- test_data/s2mel_outputs/ - S2Mel (LengthRegulator, DiT, WaveNet, CFM) outputs
- test_data/bigvgan_v2/    - BigVGAN vocoder outputs

Usage:
    cd ~/Projects/index-tts
    python ~/Projects/mlx-indextts/scripts/dump_pytorch_outputs_v2.py

    # Specific modules only:
    python ~/Projects/mlx-indextts/scripts/dump_pytorch_outputs_v2.py --module gpt
    python ~/Projects/mlx-indextts/scripts/dump_pytorch_outputs_v2.py --module s2mel
    python ~/Projects/mlx-indextts/scripts/dump_pytorch_outputs_v2.py --module bigvgan
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
import librosa

# Setup paths
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

# Output directories
BASE_OUTPUT_DIR = Path.home() / "Projects" / "mlx-indextts" / "test_data"
GPT_V2_DIR = BASE_OUTPUT_DIR / "gpt_v2"
S2MEL_DIR = BASE_OUTPUT_DIR / "s2mel_outputs"
BIGVGAN_DIR = BASE_OUTPUT_DIR / "bigvgan_v2"


def save_tensor(output_dir: Path, name: str, tensor):
    """Save tensor as numpy file."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.array(tensor)
    path = output_dir / f"{name}.npy"
    np.save(path, arr)
    print(f"  Saved {name}: {arr.shape}")


def dump_gpt_v2_outputs(device: str):
    """Dump GPT v2 conditioning outputs."""
    print("\n" + "=" * 60)
    print("Dumping GPT v2 Outputs")
    print("=" * 60)

    GPT_V2_DIR.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf
    from indextts.infer_v2 import IndexTTS2
    from transformers import SeamlessM4TFeatureExtractor

    # Load model
    model_dir = "indexTTS-2"
    cfg_path = os.path.join(model_dir, "config.yaml")
    cfg = OmegaConf.load(cfg_path)

    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_cuda_kernel=False,
        device=device,
    )

    # Test inputs
    ref_audio_path = Path.home() / "Projects" / "mlx-indextts" / "ref_audios" / "voice_01.wav"

    # Load and process audio
    audio, sr = librosa.load(str(ref_audio_path), sr=None)
    audio = torch.tensor(audio).unsqueeze(0)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

    # Extract W2V-BERT features
    extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    inputs = extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get semantic embedding
    from indextts.utils.maskgct_utils import build_semantic_model
    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        os.path.join(model_dir, cfg.w2v_stat))
    semantic_model = semantic_model.to(device).eval()
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)

    with torch.no_grad():
        vq_emb = semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        spk_cond_emb = vq_emb.hidden_states[17]
        spk_cond_emb = (spk_cond_emb - semantic_mean) / semantic_std

    # Save inputs
    # speech_condition is spk_cond_emb transposed to NCL
    speech_condition = spk_cond_emb.transpose(1, 2)  # (B, 1024, T) NCL
    save_tensor(GPT_V2_DIR, "speech_condition", speech_condition)
    save_tensor(GPT_V2_DIR, "emo_speech_condition", speech_condition)  # Same for emo

    # Get GPT conditioning outputs
    cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=device)

    with torch.no_grad():
        # Speaker conditioning
        spk_cond = tts.gpt.get_conditioning(speech_condition, cond_lengths)
        save_tensor(GPT_V2_DIR, "spk_conditioning", spk_cond)

        # Emotion conditioning (raw)
        emo_vec_raw = tts.gpt.get_emo_conditioning(speech_condition, cond_lengths)
        save_tensor(GPT_V2_DIR, "emo_vec_raw", emo_vec_raw)

        # Emotion vector (after layers)
        emo_vec = tts.gpt.emovec_layer(emo_vec_raw)
        emo_vec = tts.gpt.emo_layer(emo_vec)
        save_tensor(GPT_V2_DIR, "emo_vec", emo_vec)

        # Conditioning latent
        use_speed = torch.zeros(1).to(device).long()
        cond_latent = tts.gpt.prepare_conditioning_latents(spk_cond, emo_vec, use_speed)
        save_tensor(GPT_V2_DIR, "conditioning_latent", cond_latent)

        # Text tokens for prepare_inputs test
        text = "你好"
        text_tokens_list = tts.tokenizer.tokenize(text)
        text_token_ids = tts.tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_tokens = torch.tensor(text_token_ids, dtype=torch.int32, device=device).unsqueeze(0)
        save_tensor(GPT_V2_DIR, "text_tokens", text_tokens)

        # Prepare inputs
        inputs_embeds, _ = tts.gpt.prepare_inputs(cond_latent, text_tokens)
        save_tensor(GPT_V2_DIR, "inputs_embeds", inputs_embeds)

    print(f"GPT v2 outputs saved to {GPT_V2_DIR}")


def dump_s2mel_outputs(device: str):
    """Dump S2Mel intermediate outputs."""
    print("\n" + "=" * 60)
    print("Dumping S2Mel Outputs")
    print("=" * 60)

    S2MEL_DIR.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf
    from huggingface_hub import hf_hub_download
    import safetensors.torch

    # Load config
    cfg_path = "indexTTS-2/config.yaml"
    cfg = OmegaConf.load(cfg_path)
    model_dir = "indexTTS-2"

    # Load S2Mel
    from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
    from indextts.s2mel.modules.audio import mel_spectrogram

    s2mel_path = os.path.join(model_dir, cfg.s2mel_checkpoint)
    s2mel = MyModel(cfg.s2mel, use_gpt_latent=True)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel, None, s2mel_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    s2mel = s2mel.to(device).eval()
    s2mel.models['cfm'].estimator.setup_caches(max_batch_size=2, max_seq_length=8192)
    print(f"S2Mel loaded from: {s2mel_path}")

    # Load Semantic Codec
    from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec

    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        os.path.join(model_dir, cfg.w2v_stat))
    semantic_model = semantic_model.to(device).eval()
    semantic_mean = semantic_mean.to(device)
    semantic_std = semantic_std.to(device)

    semantic_codec = build_semantic_codec(cfg.semantic_codec)
    semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
    safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
    semantic_codec = semantic_codec.to(device).eval()

    # Load CAMPPlus
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = hf_hub_download("funasr/campplus", filename="campplus_cn_common.bin")
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model = campplus_model.to(device).eval()

    # Load W2V-BERT
    from transformers import SeamlessM4TFeatureExtractor
    extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    # Mel function
    mel_fn_args = {
        "n_fft": cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
        "win_size": cfg.s2mel['preprocess_params']['spect_params']['win_length'],
        "hop_size": cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
        "num_mels": cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": cfg.s2mel["preprocess_params"]["sr"],
        "fmin": cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

    # Prepare test data
    ref_audio_path = Path.home() / "Projects" / "mlx-indextts" / "ref_audios" / "voice_01.wav"

    audio, sr = librosa.load(str(ref_audio_path), sr=None)
    audio = torch.tensor(audio).unsqueeze(0)
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

    # Get semantic features
    inputs = extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        vq_emb = semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        spk_cond_emb = vq_emb.hidden_states[17]
        spk_cond_emb = (spk_cond_emb - semantic_mean) / semantic_std

    # Get semantic codes
    with torch.no_grad():
        _, S_ref = semantic_codec.quantize(spk_cond_emb)

    # Get reference mel
    ref_mel = mel_fn(audio_22k.to(device).float())
    ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(device)

    # Get style (CAMPPlus)
    feat = torchaudio.compliance.kaldi.fbank(
        audio_16k.to(device),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    with torch.no_grad():
        style = campplus_model(feat.unsqueeze(0))

    print(f"spk_cond_emb: {spk_cond_emb.shape}")
    print(f"S_ref: {S_ref.shape}")
    print(f"ref_mel: {ref_mel.shape}")
    print(f"style: {style.shape}")

    # === LengthRegulator (prompt condition) ===
    print("\n--- LengthRegulator (prompt_condition) ---")
    with torch.no_grad():
        prompt_condition, _, _, _, _ = s2mel.models['length_regulator'](
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
        )
    save_tensor(S2MEL_DIR, "length_reg_input_S_ref", S_ref)
    save_tensor(S2MEL_DIR, "length_reg_input_ylens", ref_target_lengths)
    save_tensor(S2MEL_DIR, "length_reg_output_prompt_cond", prompt_condition)

    # === gpt_layer ===
    print("\n--- gpt_layer ---")
    fake_gpt_latent = torch.randn(1, S_ref.shape[1], 1280, device=device)
    with torch.no_grad():
        latent = s2mel.models['gpt_layer'](fake_gpt_latent)
    save_tensor(S2MEL_DIR, "gpt_layer_input", fake_gpt_latent)
    save_tensor(S2MEL_DIR, "gpt_layer_output", latent)

    # === LengthRegulator (inference condition) ===
    print("\n--- LengthRegulator (inference condition) ---")
    codes = S_ref[:, :, 0].long()
    codes = torch.clamp(codes, 0, 8193)
    code_lens = torch.tensor([codes.shape[1]], device=device)

    with torch.no_grad():
        S_infer = semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
        S_infer = S_infer.transpose(1, 2)
        S_infer = S_infer + latent

    target_lengths = (code_lens * 1.72).long()
    with torch.no_grad():
        cond, _, _, _, _ = s2mel.models['length_regulator'](
            S_infer, ylens=target_lengths, n_quantizers=3, f0=None
        )
    save_tensor(S2MEL_DIR, "length_reg_infer_input", S_infer)
    save_tensor(S2MEL_DIR, "length_reg_infer_ylens", target_lengths)
    save_tensor(S2MEL_DIR, "length_reg_infer_output", cond)

    # === CFM inference with hooks ===
    print("\n--- CFM inference ---")
    cat_condition = torch.cat([prompt_condition, cond], dim=1)
    save_tensor(S2MEL_DIR, "cfm_input_mu", cat_condition)
    save_tensor(S2MEL_DIR, "cfm_input_x_lens", np.array([cat_condition.size(1)]))
    save_tensor(S2MEL_DIR, "cfm_input_prompt", ref_mel)
    save_tensor(S2MEL_DIR, "cfm_input_style", style)

    # Hook to capture DiT inputs/outputs
    dit_inputs = {}
    dit_outputs = {}

    def hook_dit_forward(module, inputs, output):
        dit_inputs['x'] = inputs[0].detach().cpu().numpy()
        dit_inputs['prompt_x'] = inputs[1].detach().cpu().numpy()
        dit_inputs['x_lens'] = inputs[2].detach().cpu().numpy()
        dit_inputs['t'] = inputs[3].detach().cpu().numpy()
        dit_inputs['style'] = inputs[4].detach().cpu().numpy()
        dit_inputs['cond'] = inputs[5].detach().cpu().numpy()
        dit_outputs['output'] = output.detach().cpu().numpy()

    estimator = s2mel.models['cfm'].estimator
    hook = estimator.register_forward_hook(hook_dit_forward)

    with torch.no_grad():
        vc_target = s2mel.models['cfm'].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(device),
            ref_mel, style, None,
            n_timesteps=25,
            inference_cfg_rate=0.7
        )

    hook.remove()

    # Save DiT inputs/outputs
    for key, val in dit_inputs.items():
        save_tensor(S2MEL_DIR, f"dit_input_{key}", val)
    save_tensor(S2MEL_DIR, "dit_output", dit_outputs['output'])

    save_tensor(S2MEL_DIR, "cfm_output_full", vc_target)
    save_tensor(S2MEL_DIR, "cfm_output_trimmed", vc_target[:, :, ref_mel.size(-1):])

    # === DiT submodule exports ===
    print("\n--- DiT submodules ---")
    dit = estimator
    B, _, T = dit_inputs['x'].shape

    x = torch.tensor(dit_inputs['x']).to(device)
    prompt_x = torch.tensor(dit_inputs['prompt_x']).to(device)
    t = torch.tensor(dit_inputs['t']).to(device)
    style_t = torch.tensor(dit_inputs['style']).to(device)
    cond_t = torch.tensor(dit_inputs['cond']).to(device)

    with torch.no_grad():
        t1 = dit.t_embedder(t)
        save_tensor(S2MEL_DIR, "dit_t_embedder_output", t1)

        cond_proj = dit.cond_projection(cond_t)
        save_tensor(S2MEL_DIR, "dit_cond_projection_output", cond_proj)

        x_t = x.transpose(1, 2)
        prompt_x_t = prompt_x.transpose(1, 2)
        x_in = torch.cat([x_t, prompt_x_t, cond_proj], dim=-1)
        if dit.transformer_style_condition and not dit.style_as_token:
            x_in = torch.cat([x_in, style_t[:, None, :].repeat(1, T, 1)], dim=-1)
        save_tensor(S2MEL_DIR, "dit_cond_x_merge_input", x_in)

        x_merged = dit.cond_x_merge_linear(x_in)
        save_tensor(S2MEL_DIR, "dit_cond_x_merge_output", x_merged)

    # === WaveNet intermediate ===
    print("\n--- WaveNet intermediate ---")
    with torch.no_grad():
        input_pos = dit.input_pos[:x_merged.size(1)]
        x_mask = torch.ones(1, 1, x_merged.size(1), device=device).bool()
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_merged.size(1), 1)

        x_res = dit.transformer(x_merged, t1.unsqueeze(1), input_pos, x_mask_expanded)

        if dit.long_skip_connection:
            x_res = dit.skip_linear(torch.cat([x_res, x_t], dim=-1))

        save_tensor(S2MEL_DIR, "wavenet_x_res_input", x_res)

        x_wn = dit.conv1(x_res).transpose(1, 2)
        save_tensor(S2MEL_DIR, "wavenet_conv1_output", x_wn)

        t2 = dit.t_embedder2(t)
        save_tensor(S2MEL_DIR, "wavenet_t2_input", t2)

        x_mask_wn = torch.ones(1, 1, x_wn.size(2), device=device)
        wn_out = dit.wavenet(x_wn, x_mask_wn, g=t2.unsqueeze(2))
        save_tensor(S2MEL_DIR, "wavenet_output", wn_out)

    print(f"S2Mel outputs saved to {S2MEL_DIR}")


def dump_bigvgan_v2_outputs(device: str):
    """Dump BigVGAN v2 intermediate outputs."""
    print("\n" + "=" * 60)
    print("Dumping BigVGAN v2 Outputs")
    print("=" * 60)

    BIGVGAN_DIR.mkdir(parents=True, exist_ok=True)

    # Load BigVGAN from HuggingFace
    from huggingface_hub import hf_hub_download
    import json

    # Download BigVGAN v2
    config_path = hf_hub_download("nvidia/bigvgan_v2_22khz_80band_256x", "config.json")
    weights_path = hf_hub_download("nvidia/bigvgan_v2_22khz_80band_256x", "bigvgan_generator.pt")

    with open(config_path) as f:
        config = json.load(f)

    # Load model
    sys.path.insert(0, str(Path.home() / "Projects/index-tts"))
    from indextts.bigvgan import BigVGAN as BigVGANPT

    model = BigVGANPT(
        num_mels=config["num_mels"],
        upsample_rates=config["upsample_rates"],
        upsample_kernel_sizes=config["upsample_kernel_sizes"],
        upsample_initial_channel=config["upsample_initial_channel"],
        resblock_kernel_sizes=config["resblock_kernel_sizes"],
        resblock_dilation_sizes=config["resblock_dilation_sizes"],
        resblock=config["resblock"],
        activation=config["activation"],
        snake_logscale=config.get("snake_logscale", True),
        use_tanh_at_final=config.get("use_tanh_at_final", False),
        use_bias_at_final=config.get("use_bias_at_final", False),
    )
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict["generator"])
    model = model.to(device).eval()

    # Create test input
    np.random.seed(42)
    mel_input = np.random.randn(1, 80, 100).astype(np.float32) * 3.0
    mel_pt = torch.tensor(mel_input).to(device)

    save_tensor(BIGVGAN_DIR, "mel_input", mel_input)

    # Run forward with hooks
    intermediate = {}

    def hook_conv_pre(m, inp, out):
        intermediate['conv_pre'] = out.detach().cpu().numpy()

    def hook_ups0(m, inp, out):
        intermediate['ups0'] = out.detach().cpu().numpy()

    def hook_resblocks0(m, inp, out):
        intermediate['resblocks0'] = out.detach().cpu().numpy()

    model.conv_pre.register_forward_hook(hook_conv_pre)
    model.ups[0].register_forward_hook(hook_ups0)
    model.resblocks[0].register_forward_hook(hook_resblocks0)

    with torch.no_grad():
        wav = model(mel_pt)

    save_tensor(BIGVGAN_DIR, "step1_conv_pre", intermediate['conv_pre'])
    save_tensor(BIGVGAN_DIR, "step2_ups0", intermediate['ups0'])
    save_tensor(BIGVGAN_DIR, "step3_resblocks0", intermediate['resblocks0'])
    save_tensor(BIGVGAN_DIR, "final_wav", wav)

    print(f"BigVGAN v2 outputs saved to {BIGVGAN_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Dump PyTorch outputs for IndexTTS 2.0")
    parser.add_argument("--module", choices=["gpt", "s2mel", "bigvgan", "all"], default="all",
                        help="Which module to dump")
    args = parser.parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if args.module in ["gpt", "all"]:
        dump_gpt_v2_outputs(device)

    if args.module in ["s2mel", "all"]:
        dump_s2mel_outputs(device)

    if args.module in ["bigvgan", "all"]:
        dump_bigvgan_v2_outputs(device)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Outputs saved to {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
