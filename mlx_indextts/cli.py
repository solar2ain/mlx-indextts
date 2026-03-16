"""Command-line interface for MLX-IndexTTS."""

import argparse
import sys
from pathlib import Path


def convert_command(args):
    """Handle the convert command."""
    from mlx_indextts.convert import convert_model

    convert_model(
        model_dir=args.model_dir,
        output_dir=args.output,
        config_path=args.config,
    )


def generate_command(args):
    """Handle the generate command."""
    from mlx_indextts.generate import IndexTTS
    import subprocess
    import sys

    # Load model
    print(f"Loading model from {args.model}...")
    tts = IndexTTS.load_model(args.model)

    # Generate speech
    print(f"Generating speech for: {args.text[:50]}...")
    audio = tts.generate(
        text=args.text,
        ref_audio=args.ref_audio,
        max_mel_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Save output
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
    tts = IndexTTS.load_model(args.model)

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
        description="IndexTTS for Apple Silicon using MLX",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert PyTorch model to MLX format",
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
    convert_parser.set_defaults(func=convert_command)

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate speech from text",
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
        required=True,
        help="Output audio file path",
    )
    generate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum mel tokens to generate (default: 600)",
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
        help="Random seed for reproducible generation",
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
    generate_parser.set_defaults(func=generate_command)

    # Speaker command (save pre-computed conditioning)
    speaker_parser = subparsers.add_parser(
        "speaker",
        help="Save pre-computed speaker conditioning for faster inference",
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
