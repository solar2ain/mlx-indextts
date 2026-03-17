"""Command-line interface for MLX-IndexTTS."""

import argparse
import sys
from pathlib import Path


def detect_pytorch_version(model_dir: Path) -> str:
    """Detect IndexTTS version from PyTorch model directory.

    Returns '2.0' if s2mel.pth exists, otherwise '1.5'.
    """
    if (model_dir / "s2mel.pth").exists():
        return "2.0"
    return "1.5"


def detect_mlx_version(model_dir: Path) -> str:
    """Detect IndexTTS version from MLX model directory.

    Returns '2.0' if s2mel.safetensors exists, otherwise '1.5'.
    """
    if (model_dir / "s2mel.safetensors").exists():
        return "2.0"
    return "1.5"


def convert_command(args):
    """Handle the convert command (auto-detect version)."""
    model_dir = Path(args.model_dir)
    version = detect_pytorch_version(model_dir)

    print(f"Detected IndexTTS version: {version}")

    if version == "2.0":
        from mlx_indextts.convert_v2 import convert_model as convert_model_v2
        convert_model_v2(
            model_dir=args.model_dir,
            output_dir=args.output,
            config_path=args.config,
        )
    else:
        from mlx_indextts.convert import convert_model

        # Parse quantize option
        quantize_bits = None
        if args.quantize:
            if args.quantize.lower() == "fp32":
                quantize_bits = None
            else:
                quantize_bits = int(args.quantize)

        convert_model(
            model_dir=args.model_dir,
            output_dir=args.output,
            config_path=args.config,
            quantize_bits=quantize_bits,
        )


def generate_command(args):
    """Handle the generate command (auto-detect version)."""
    import subprocess

    model_dir = Path(args.model)
    version = detect_mlx_version(model_dir)

    # Check for v2-only parameters used with v1.5 model
    v2_only_params = []
    if getattr(args, 'emotion', None) is not None:
        v2_only_params.append('--emotion')
    if getattr(args, 'emo_alpha', 1.0) != 1.0:
        v2_only_params.append('--emo-alpha')
    if getattr(args, 'steps', 25) != 25:
        v2_only_params.append('--steps')
    if getattr(args, 'cfg', 0.7) != 0.7:
        v2_only_params.append('--cfg')

    if version == "1.5" and v2_only_params:
        print(f"Error: Parameters {v2_only_params} are only available for IndexTTS 2.0 models.")
        print(f"Detected model version: 1.5")
        sys.exit(1)

    # Default max_tokens based on version (use config defaults)
    # v1.5: 800, v2.0: 1500 (model supports up to 1815)
    if args.max_tokens is not None:
        max_tokens = args.max_tokens
    else:
        max_tokens = 1500 if version == "2.0" else 800

    # Default memory limit based on version
    memory_limit = args.memory_limit
    if memory_limit is None:
        memory_limit = 12.0 if version == "2.0" else 8.0

    print(f"Using IndexTTS {version}")

    if version == "2.0":
        from mlx_indextts.generate_v2 import IndexTTSv2

        tts = IndexTTSv2(
            model_dir=args.model,
            memory_limit_gb=memory_limit,
        )

        audio = tts.generate(
            text=args.text,
            reference_audio=args.ref_audio,
            output_path=args.output,
            max_mel_tokens=max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            diffusion_steps=getattr(args, 'steps', 25),
            cfg_rate=getattr(args, 'cfg', 0.7),
            emotion=getattr(args, 'emotion', None),
            emo_alpha=getattr(args, 'emo_alpha', 1.0),
            seed=getattr(args, 'seed', None),
            verbose=args.verbose,
        )
    else:
        from mlx_indextts.generate import IndexTTS

        # Parse quantize option
        quantize_bits = None
        if args.quantize:
            if args.quantize.lower() == "fp32":
                quantize_bits = None
            else:
                quantize_bits = int(args.quantize)

        tts = IndexTTS.load_model(
            args.model,
            memory_limit_gb=memory_limit,
            quantize_bits=quantize_bits,
        )

        print(f"Generating speech for: {args.text[:50]}...")
        audio = tts.generate(
            text=args.text,
            ref_audio=args.ref_audio,
            max_mel_tokens=max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=getattr(args, 'seed', None),
            verbose=args.verbose,
        )

        tts.save_audio(audio, args.output)

    print(f"Audio saved to {args.output}")

    # Play audio if requested
    if args.play:
        if sys.platform == "darwin":
            subprocess.run(["afplay", args.output])
        elif sys.platform == "linux":
            subprocess.run(["aplay", args.output])
        else:
            print("Auto-play not supported on this platform")


def speaker_command(args):
    """Handle the speaker command - save pre-computed speaker conditioning."""
    from mlx_indextts.generate import IndexTTS
    import time

    # Load model
    print(f"Loading model from {args.model}...")
    tts = IndexTTS.load_model(args.model, memory_limit_gb=args.memory_limit)

    # Compute and save speaker
    print(f"Computing speaker conditioning from {args.ref_audio}...")
    start = time.perf_counter()
    tts.save_speaker(args.ref_audio, args.output)
    elapsed = time.perf_counter() - start
    print(f"Speaker saved to {args.output} ({elapsed:.2f}s)")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="mlx-indextts",
        description="IndexTTS for Apple Silicon using MLX (supports v1.5 and v2.0)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command (auto-detect version)
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert PyTorch model to MLX format (auto-detects v1.5/v2.0)",
    )
    convert_parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing PyTorch checkpoints",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for MLX weights",
    )
    convert_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: model_dir/config.yaml)",
    )
    convert_parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        default="fp32",
        help="Quantization bits: 4, 8, or fp32 (v1.5 only, default: fp32)",
    )
    convert_parser.set_defaults(func=convert_command)

    # Generate command (auto-detect version)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate speech from text (auto-detects v1.5/v2.0)",
    )
    generate_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to converted MLX model directory",
    )
    generate_parser.add_argument(
        "--ref-audio",
        "-r",
        type=str,
        required=True,
        help="Reference audio file for voice cloning",
    )
    generate_parser.add_argument(
        "--text",
        "-t",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.wav",
        help="Output audio file path (default: output.wav)",
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum mel tokens to generate (default: 800)",
    )
    generate_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    generate_parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Top-k sampling (default: 30)",
    )
    generate_parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling (default: 0.8)",
    )
    generate_parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="随机种子，用于可复现生成",
    )
    generate_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    generate_parser.add_argument(
        "--play",
        "-p",
        action="store_true",
        help="Play audio after generation (macOS/Linux)",
    )
    generate_parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help="GPU 内存限制 GB (default: 8 for v1.5, 12 for v2.0)",
    )
    generate_parser.add_argument(
        "--quantize",
        "-q",
        type=str,
        default="fp32",
        help="[v1.5 only] 运行时量化: 4, 8, 或 fp32 (default: fp32)",
    )
    # v2.0 specific options
    generate_parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="[v2.0 only] S2Mel diffusion/CFM 采样步数，越大质量越高但越慢 (default: 25)",
    )
    generate_parser.add_argument(
        "--cfg",
        type=float,
        default=0.7,
        help="[v2.0 only] Classifier-Free Guidance 强度，控制生成与条件的一致性 (default: 0.7)",
    )
    generate_parser.add_argument(
        "--emotion",
        type=str,
        default=None,
        help="[v2.0 only] 情感控制: happy/sad/angry/afraid/disgusted/melancholic/surprised/calm，或 'happy:0.8,sad:0.2'",
    )
    generate_parser.add_argument(
        "--emo-alpha",
        type=float,
        default=1.0,
        help="[v2.0 only] 情感强度 0.0-1.0，0=参考音频情感，1=完全指定情感 (default: 1.0)",
    )
    generate_parser.set_defaults(func=generate_command)

    # Speaker command (save pre-computed conditioning, v1.5 only)
    speaker_parser = subparsers.add_parser(
        "speaker",
        help="Save pre-computed speaker conditioning for faster inference (v1.5 only)",
    )
    speaker_parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Path to converted MLX model directory",
    )
    speaker_parser.add_argument(
        "--ref-audio",
        "-r",
        type=str,
        required=True,
        help="Reference audio file",
    )
    speaker_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output .npz file path for speaker conditioning",
    )
    speaker_parser.add_argument(
        "--memory-limit",
        type=float,
        default=8.0,
        help="GPU memory limit in GB (default: 8.0, 0 for no limit)",
    )
    speaker_parser.set_defaults(func=speaker_command)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
