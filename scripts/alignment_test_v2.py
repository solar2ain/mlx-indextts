#!/usr/bin/env python3
"""
IndexTTS 2.0 MLX vs PyTorch Alignment Test

This script tests alignment between MLX and PyTorch implementations
for all major modules in IndexTTS 2.0:
- GPT v2 (conditioning, emotion, input preparation)
- S2Mel (GPTLayer, LengthRegulator, DiT, WaveNet, CFM)
- BigVGAN v2 (vocoder)

Usage:
    # Run all tests
    uv run python scripts/alignment_test_v2.py

    # Run specific module tests
    uv run python scripts/alignment_test_v2.py --module gpt
    uv run python scripts/alignment_test_v2.py --module s2mel
    uv run python scripts/alignment_test_v2.py --module bigvgan
    uv run python scripts/alignment_test_v2.py --module cfm

Prerequisites:
    1. Export PyTorch outputs first:
       uv run python scripts/dump_pytorch_outputs_v2.py

    2. Convert models:
       uv run python -m mlx_indextts.convert_v2 --model-dir ~/Projects/index-tts/indexTTS-2 -o models/mlx-indexTTS-2.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx


# Test data directories
TEST_DATA_DIR = Path("test_data")
GPT_V2_DIR = TEST_DATA_DIR / "gpt_v2"
S2MEL_DIR = TEST_DATA_DIR / "s2mel_outputs"
BIGVGAN_DIR = TEST_DATA_DIR / "bigvgan_v2"

# Model directory
MODEL_DIR = Path("models/mlx-indexTTS-2.0")


def compare(name: str, mlx_arr: np.ndarray, pt_arr: np.ndarray, atol: float = 0.001) -> bool:
    """Compare MLX and PyTorch arrays, report MAE and status."""
    if mlx_arr.shape != pt_arr.shape:
        print(f"  SHAPE MISMATCH {name}: MLX {mlx_arr.shape} vs PT {pt_arr.shape}")
        return False

    mae = np.mean(np.abs(mlx_arr - pt_arr))
    max_diff = np.max(np.abs(mlx_arr - pt_arr))

    status = "PASS" if mae < atol else "FAIL"
    symbol = "+" if mae < atol else "-"
    print(f"  [{symbol}] {name}: MAE={mae:.6f}, max_diff={max_diff:.4f}")

    return mae < atol


def test_gpt_v2() -> dict:
    """Test GPT v2 alignment."""
    print("\n" + "=" * 60)
    print("GPT v2 Alignment Test")
    print("=" * 60)

    results = {}

    # Check test data exists
    if not GPT_V2_DIR.exists():
        print(f"Error: Test data not found at {GPT_V2_DIR}")
        print("Run: uv run python scripts/dump_pytorch_outputs_v2.py")
        return {"error": "test_data_missing"}

    # Load test data
    speech_condition = np.load(GPT_V2_DIR / "speech_condition.npy")
    emo_speech_condition = np.load(GPT_V2_DIR / "emo_speech_condition.npy")
    text_tokens = np.load(GPT_V2_DIR / "text_tokens.npy")
    ref_spk_cond = np.load(GPT_V2_DIR / "spk_conditioning.npy")
    ref_emo_vec_raw = np.load(GPT_V2_DIR / "emo_vec_raw.npy")
    ref_emo_vec = np.load(GPT_V2_DIR / "emo_vec.npy")
    ref_conditioning_latent = np.load(GPT_V2_DIR / "conditioning_latent.npy")
    ref_inputs_embeds = np.load(GPT_V2_DIR / "inputs_embeds.npy")

    print(f"Loaded test data: speech_condition {speech_condition.shape}")

    # Load MLX model
    from mlx_indextts.config import IndexTTSConfig
    from mlx_indextts.models.gpt_v2 import UnifiedVoiceV2

    config_path = MODEL_DIR / "config.yaml"
    if not config_path.exists():
        config_path = Path.home() / "Projects/index-tts/indexTTS-2/config.yaml"
    config = IndexTTSConfig.from_yaml(str(config_path))
    model = UnifiedVoiceV2(config)

    weights_path = MODEL_DIR / "gpt.safetensors"
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return {"error": "weights_missing"}

    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()))
    model.eval()
    mx.eval(model.parameters())

    # Convert inputs
    speech_mx = mx.array(speech_condition)
    emo_speech_mx = mx.array(emo_speech_condition)
    text_mx = mx.array(text_tokens)
    cond_lengths = mx.array([speech_condition.shape[2]])

    # Test get_conditioning
    spk_cond = model.get_conditioning(speech_mx, cond_lengths)
    mx.eval(spk_cond)
    results["get_conditioning"] = compare("get_conditioning", np.array(spk_cond), ref_spk_cond)

    # Test get_emo_conditioning
    emo_vec_raw = model.get_emo_conditioning(emo_speech_mx, cond_lengths)
    mx.eval(emo_vec_raw)
    results["get_emo_conditioning"] = compare("get_emo_conditioning", np.array(emo_vec_raw), ref_emo_vec_raw)

    # Test emo_layer pipeline
    emo_vec = model.emovec_layer(emo_vec_raw)
    emo_vec = model.emo_layer(emo_vec)
    mx.eval(emo_vec)
    results["emo_layer_pipeline"] = compare("emo_layer_pipeline", np.array(emo_vec), ref_emo_vec)

    # Test prepare_conditioning_latents
    conds_latent = model.prepare_conditioning_latents(spk_cond, emo_vec, batch_size=1)
    mx.eval(conds_latent)
    results["prepare_conditioning_latents"] = compare(
        "prepare_conditioning_latents", np.array(conds_latent), ref_conditioning_latent
    )

    # Test prepare_inputs
    inputs_embeds, _ = model.prepare_inputs(conds_latent, text_mx)
    mx.eval(inputs_embeds)
    results["prepare_inputs"] = compare("prepare_inputs", np.array(inputs_embeds), ref_inputs_embeds)

    return results


def test_s2mel() -> dict:
    """Test S2Mel alignment."""
    print("\n" + "=" * 60)
    print("S2Mel Alignment Test")
    print("=" * 60)

    results = {}

    # Check test data
    if not S2MEL_DIR.exists():
        print(f"Error: Test data not found at {S2MEL_DIR}")
        print("Run: uv run python scripts/dump_pytorch_outputs_v2.py")
        return {"error": "test_data_missing"}

    # Load all test data
    pt_outputs = {}
    for f in S2MEL_DIR.glob("*.npy"):
        pt_outputs[f.stem] = np.load(f)

    print(f"Loaded {len(pt_outputs)} test outputs")

    # Load MLX model
    from mlx_indextts.models.s2mel import S2Mel, InterpolateRegulator

    weights_path = MODEL_DIR / "s2mel.safetensors"
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return {"error": "weights_missing"}

    s2mel = S2Mel()
    s2mel.load_weights(str(weights_path), strict=False)
    s2mel.eval()

    # Test GPTLayer
    if "gpt_layer_input" in pt_outputs:
        x_input = pt_outputs["gpt_layer_input"]
        pt_output = pt_outputs["gpt_layer_output"]

        x_mx = mx.array(x_input)
        out = s2mel.gpt_layer(x_mx)
        mx.eval(out)
        results["GPTLayer"] = compare("GPTLayer", np.array(out), pt_output)

    # Test LengthRegulator
    if "length_reg_infer_input" in pt_outputs:
        x_input = pt_outputs["length_reg_infer_input"]
        ylens = pt_outputs["length_reg_infer_ylens"]
        pt_output = pt_outputs["length_reg_infer_output"]

        # Create separate LR with correct weights
        lr = InterpolateRegulator(
            channels=512,
            sampling_ratios=(1, 1, 1, 1),
            is_discrete=False,
            in_channels=1024,
            codebook_size=2048,
            n_codebooks=1,
        )
        weights = mx.load(str(weights_path))
        lr_weights = {k.replace("length_regulator.", ""): v for k, v in weights.items() if k.startswith("length_regulator.")}
        lr.load_weights(lr_weights, strict=True)

        x_mx = mx.array(x_input)
        ylens_mx = mx.array(ylens)
        out, _, _, _, _ = lr(x_mx, ylens_mx, n_quantizers=3)
        mx.eval(out)
        results["LengthRegulator"] = compare("LengthRegulator", np.array(out), pt_output)

    # Test TimestepEmbedder
    if "dit_input_t" in pt_outputs and "dit_t_embedder_output" in pt_outputs:
        t_input = pt_outputs["dit_input_t"]
        pt_output = pt_outputs["dit_t_embedder_output"]

        t_mx = mx.array(t_input)
        out = s2mel.cfm.estimator.t_embedder(t_mx)
        mx.eval(out)
        results["TimestepEmbedder"] = compare("TimestepEmbedder", np.array(out), pt_output)

    # Test cond_projection
    if "dit_input_cond" in pt_outputs and "dit_cond_projection_output" in pt_outputs:
        cond_input = pt_outputs["dit_input_cond"]
        pt_output = pt_outputs["dit_cond_projection_output"]

        cond_mx = mx.array(cond_input)
        out = s2mel.cfm.estimator.cond_projection(cond_mx)
        mx.eval(out)
        results["cond_projection"] = compare("cond_projection", np.array(out), pt_output)

    # Test cond_x_merge_linear
    if "dit_cond_x_merge_input" in pt_outputs and "dit_cond_x_merge_output" in pt_outputs:
        merge_input = pt_outputs["dit_cond_x_merge_input"]
        pt_output = pt_outputs["dit_cond_x_merge_output"]

        merge_mx = mx.array(merge_input)
        out = s2mel.cfm.estimator.cond_x_merge_linear(merge_mx)
        mx.eval(out)
        results["cond_x_merge_linear"] = compare("cond_x_merge_linear", np.array(out), pt_output)

    # Test WaveNet components
    if "wavenet_conv1_output" in pt_outputs and "wavenet_x_res_input" in pt_outputs:
        x_res = pt_outputs["wavenet_x_res_input"]  # (2, 361, 512) NLC
        pt_conv1 = pt_outputs["wavenet_conv1_output"]  # (2, 512, 361) NCL

        x_mx = mx.array(x_res)
        conv1_out = s2mel.cfm.estimator.conv1(x_mx)
        conv1_out_ncl = conv1_out.transpose(0, 2, 1)
        mx.eval(conv1_out_ncl)
        results["WaveNet_conv1"] = compare("WaveNet_conv1", np.array(conv1_out_ncl), pt_conv1)

    if "wavenet_conv1_output" in pt_outputs and "wavenet_output" in pt_outputs and "wavenet_t2_input" in pt_outputs:
        pt_conv1 = pt_outputs["wavenet_conv1_output"]  # NCL
        t2_input = pt_outputs["wavenet_t2_input"]
        pt_wn_out = pt_outputs["wavenet_output"]

        conv1_mx = mx.array(pt_conv1)
        t2_mx = mx.array(t2_input)
        B, _, T = conv1_mx.shape
        x_mask = mx.ones((B, 1, T))
        wn_out = s2mel.cfm.estimator.wavenet(conv1_mx, x_mask, g=t2_mx[:, :, None])
        mx.eval(wn_out)
        results["WaveNet"] = compare("WaveNet", np.array(wn_out), pt_wn_out)

    # Test DiT full forward
    if all(k in pt_outputs for k in ["dit_input_x", "dit_input_prompt_x", "dit_input_t", "dit_input_style", "dit_input_cond", "dit_output"]):
        x = pt_outputs["dit_input_x"]
        prompt_x = pt_outputs["dit_input_prompt_x"]
        x_lens = pt_outputs["dit_input_x_lens"]
        t = pt_outputs["dit_input_t"]
        style = pt_outputs["dit_input_style"]
        cond = pt_outputs["dit_input_cond"]
        pt_output = pt_outputs["dit_output"]

        x_mx = mx.array(x)
        prompt_x_mx = mx.array(prompt_x)
        x_lens_mx = mx.array(x_lens)
        t_mx = mx.array(t)
        style_mx = mx.array(style)
        cond_mx = mx.array(cond)

        try:
            output = s2mel.cfm.estimator(x_mx, prompt_x_mx, x_lens_mx, t_mx, style_mx, cond_mx)
            mx.eval(output)
            # Use larger tolerance for DiT due to numerical accumulation
            results["DiT_full"] = compare("DiT_full", np.array(output), pt_output, atol=0.05)
        except Exception as e:
            print(f"  [-] DiT_full: Error - {e}")
            results["DiT_full"] = False

    return results


def test_bigvgan_v2() -> dict:
    """Test BigVGAN v2 alignment."""
    print("\n" + "=" * 60)
    print("BigVGAN v2 Alignment Test")
    print("=" * 60)

    results = {}

    # Check test data
    if not BIGVGAN_DIR.exists():
        print(f"Error: Test data not found at {BIGVGAN_DIR}")
        print("Run: uv run python scripts/dump_pytorch_outputs_v2.py")
        return {"error": "test_data_missing"}

    # Load test data
    mel_input = np.load(BIGVGAN_DIR / "mel_input.npy")
    ref_conv_pre = np.load(BIGVGAN_DIR / "step1_conv_pre.npy")
    ref_ups0 = np.load(BIGVGAN_DIR / "step2_ups0.npy")
    ref_resblocks0 = np.load(BIGVGAN_DIR / "step3_resblocks0.npy")
    ref_wav = np.load(BIGVGAN_DIR / "final_wav.npy")

    print(f"Loaded test data: mel_input {mel_input.shape}")

    # Load MLX model
    from mlx_indextts.models.bigvgan_v2 import BigVGANV2, BigVGANV2Config

    config = BigVGANV2Config()
    model = BigVGANV2(config)

    weights_path = MODEL_DIR / "bigvgan.safetensors"
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return {"error": "weights_missing"}

    weights = mx.load(str(weights_path))
    model.load_weights(list(weights.items()))
    model.eval()

    mel_mx = mx.array(mel_input)

    # Test conv_pre
    x = mel_mx.transpose(0, 2, 1)  # NCL -> NLC
    x = model.conv_pre(x)
    x = x.transpose(0, 2, 1)  # NLC -> NCL
    mx.eval(x)
    results["conv_pre"] = compare("conv_pre", np.array(x), ref_conv_pre, atol=1e-5)

    # Test ups.0
    x = x.transpose(0, 2, 1)
    x = model.ups[0](x)
    x = x.transpose(0, 2, 1)
    mx.eval(x)
    results["ups_0"] = compare("ups_0", np.array(x), ref_ups0, atol=1e-5)

    # Test resblocks.0
    xs = model.resblocks[0](x)
    mx.eval(xs)
    results["resblocks_0"] = compare("resblocks_0", np.array(xs), ref_resblocks0)

    # Test full model
    wav = model(mel_mx)
    mx.eval(wav)
    wav_np = np.array(wav)

    min_len = min(ref_wav.shape[-1], wav_np.shape[-1])
    results["full_model"] = compare("full_model", wav_np[..., :min_len], ref_wav[..., :min_len], atol=0.01)

    # Report range
    print(f"  MLX range: [{wav_np.min():.4f}, {wav_np.max():.4f}]")
    print(f"  PT range:  [{ref_wav.min():.4f}, {ref_wav.max():.4f}]")

    return results


def test_cfm() -> dict:
    """Test CFM with identical inputs."""
    print("\n" + "=" * 60)
    print("CFM Alignment Test (identical inputs)")
    print("=" * 60)

    results = {}

    # Check dependencies
    try:
        import torch
        from omegaconf import OmegaConf
    except ImportError:
        print("Error: PyTorch or OmegaConf not available for CFM comparison")
        return {"error": "dependencies_missing"}

    # Create identical inputs
    np.random.seed(42)
    mx.random.seed(42)
    torch.manual_seed(42)

    B, C, T = 1, 80, 200
    content_dim = 512
    style_dim = 192
    prompt_len = 80

    z_np = np.random.randn(B, C, T).astype(np.float32)
    prompt_np = np.random.randn(B, C, prompt_len).astype(np.float32) * 3.0 - 5.0
    mu_np = np.random.randn(B, T, content_dim).astype(np.float32) * 0.1
    style_np = np.random.randn(B, style_dim).astype(np.float32)
    x_lens_np = np.array([T])

    print(f"Test inputs: z {z_np.shape}, prompt {prompt_np.shape}, mu {mu_np.shape}")

    # Load MLX model
    from mlx_indextts.models.s2mel import S2Mel

    weights_path = MODEL_DIR / "s2mel.safetensors"
    if not weights_path.exists():
        print(f"Error: Weights not found at {weights_path}")
        return {"error": "weights_missing"}

    s2mel_mlx = S2Mel()
    s2mel_mlx.load_weights(str(weights_path), strict=True)
    s2mel_mlx.eval()

    # MLX CFM forward
    z_mx = mx.array(z_np)
    prompt_mx = mx.array(prompt_np)
    mu_mx = mx.array(mu_np)
    style_mx = mx.array(style_np)
    x_lens_mx = mx.array(x_lens_np)

    cfm = s2mel_mlx.cfm
    x = z_mx
    t_span = mx.linspace(0, 1, 26)

    prompt_x = mx.concatenate([prompt_mx, mx.zeros((B, C, T - prompt_len))], axis=2)
    x = mx.concatenate([mx.zeros((B, C, prompt_len)), x[:, :, prompt_len:]], axis=2)

    t = t_span[0]
    inference_cfg_rate = 0.7

    for step in range(1, len(t_span)):
        dt = t_span[step] - t_span[step - 1]

        stacked_prompt_x = mx.concatenate([prompt_x, mx.zeros_like(prompt_x)], axis=0)
        stacked_style = mx.concatenate([style_mx, mx.zeros_like(style_mx)], axis=0)
        stacked_mu = mx.concatenate([mu_mx, mx.zeros_like(mu_mx)], axis=0)
        stacked_x = mx.concatenate([x, x], axis=0)
        stacked_t = mx.array([t, t])

        stacked_dphi_dt = cfm.estimator(stacked_x, stacked_prompt_x, x_lens_mx, stacked_t, stacked_style, stacked_mu)
        dphi_dt, cfg_dphi_dt = mx.split(stacked_dphi_dt, 2, axis=0)
        dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt

        x = x + dt * dphi_dt
        t = t + dt
        x = mx.concatenate([mx.zeros((B, C, prompt_len)), x[:, :, prompt_len:]], axis=2)
        mx.eval(x)

    mlx_output = np.array(x)

    # Load PyTorch model
    sys.path.insert(0, str(Path.home() / "Projects/index-tts"))
    from indextts.s2mel.modules.flow_matching import CFM as CFMPT

    cfg = OmegaConf.load(str(Path.home() / "Projects/index-tts/indexTTS-2/config.yaml"))
    cfm_pt = CFMPT(cfg.s2mel)
    cfm_pt.to("cpu")
    cfm_pt.estimator.setup_caches(max_batch_size=2, max_seq_length=T + 10)

    state_dict = torch.load(str(Path.home() / "Projects/index-tts/indexTTS-2/s2mel.pth"), map_location="cpu")
    cfm_pt.load_state_dict(state_dict["net"]["cfm"], strict=True)
    cfm_pt.eval()

    # PyTorch CFM forward
    z_pt = torch.tensor(z_np)
    prompt_pt = torch.tensor(prompt_np)
    mu_pt = torch.tensor(mu_np)
    style_pt = torch.tensor(style_np)
    x_lens_pt = torch.tensor(x_lens_np)
    x_pt = z_pt
    t_span_pt = torch.linspace(0, 1, 26)

    prompt_x_pt = torch.zeros_like(x_pt)
    prompt_x_pt[..., :prompt_len] = prompt_pt
    x_pt[..., :prompt_len] = 0

    t_pt = t_span_pt[0]

    with torch.no_grad():
        for step in range(1, len(t_span_pt)):
            dt_pt = t_span_pt[step] - t_span_pt[step - 1]

            stacked_prompt_x_pt = torch.cat([prompt_x_pt, torch.zeros_like(prompt_x_pt)], dim=0)
            stacked_style_pt = torch.cat([style_pt, torch.zeros_like(style_pt)], dim=0)
            stacked_mu_pt = torch.cat([mu_pt, torch.zeros_like(mu_pt)], dim=0)
            stacked_x_pt = torch.cat([x_pt, x_pt], dim=0)
            stacked_t_pt = torch.cat([t_pt.unsqueeze(0), t_pt.unsqueeze(0)], dim=0)

            stacked_dphi_dt_pt = cfm_pt.estimator(
                stacked_x_pt, stacked_prompt_x_pt, x_lens_pt, stacked_t_pt, stacked_style_pt, stacked_mu_pt
            )

            dphi_dt_pt, cfg_dphi_dt_pt = stacked_dphi_dt_pt.chunk(2, dim=0)
            dphi_dt_pt = (1.0 + inference_cfg_rate) * dphi_dt_pt - inference_cfg_rate * cfg_dphi_dt_pt

            x_pt = x_pt + dt_pt * dphi_dt_pt
            t_pt = t_pt + dt_pt
            x_pt[:, :, :prompt_len] = 0

    pt_output = x_pt.numpy()

    results["CFM_forward"] = compare("CFM_forward", mlx_output, pt_output, atol=0.001)

    print(f"  MLX: mean={mlx_output.mean():.4f}, std={mlx_output.std():.4f}")
    print(f"  PT:  mean={pt_output.mean():.4f}, std={pt_output.std():.4f}")

    return results


def print_summary(all_results: dict):
    """Print summary of all test results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_pass = 0
    total_fail = 0
    total_skip = 0

    for module, results in all_results.items():
        if "error" in results:
            print(f"\n{module}: SKIPPED ({results['error']})")
            total_skip += 1
            continue

        passed = sum(1 for v in results.values() if v is True)
        failed = sum(1 for v in results.values() if v is False)
        total_pass += passed
        total_fail += failed

        status = "PASS" if failed == 0 else "FAIL"
        print(f"\n{module}: {status} ({passed}/{passed+failed})")
        for name, result in results.items():
            symbol = "+" if result else "-"
            print(f"  [{symbol}] {name}")

    print("\n" + "-" * 60)
    print(f"Total: {total_pass} passed, {total_fail} failed, {total_skip} skipped")

    if total_fail == 0 and total_skip == 0:
        print("\nAll tests passed!")
    elif total_fail > 0:
        print("\nSome tests failed. Check the output above for details.")


def main():
    parser = argparse.ArgumentParser(description="IndexTTS 2.0 Alignment Test")
    parser.add_argument("--module", choices=["gpt", "s2mel", "bigvgan", "cfm", "all"], default="all",
                        help="Which module to test")
    args = parser.parse_args()

    all_results = {}

    if args.module in ["gpt", "all"]:
        all_results["GPT_v2"] = test_gpt_v2()

    if args.module in ["s2mel", "all"]:
        all_results["S2Mel"] = test_s2mel()

    if args.module in ["bigvgan", "all"]:
        all_results["BigVGAN_v2"] = test_bigvgan_v2()

    if args.module in ["cfm", "all"]:
        all_results["CFM"] = test_cfm()

    print_summary(all_results)


if __name__ == "__main__":
    main()
