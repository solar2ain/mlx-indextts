#!/usr/bin/env python3
"""Generate test audio using both MLX and PyTorch versions.

Usage:
    uv run python scripts/generate_test_audio.py

Output files will be saved to the current directory with format:
    test_{version}_{lang}.wav
"""

import subprocess
import sys
import time
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# Test texts
CHINESE_TEXT = "人工智能正在深刻改变我们的生活方式。从智能手机的语音助手，到自动驾驶汽车，再到医疗诊断系统，AI技术已经渗透到各个领域。未来，随着技术的不断进步，人工智能将会带来更多令人惊叹的创新应用。"
ENGLISH_TEXT = "Artificial intelligence is profoundly changing how we live. From voice assistants on smartphones to self-driving cars and medical diagnosis systems, AI technology has permeated every field. In the future, as technology continues to advance, artificial intelligence will bring even more amazing innovations."

# Paths
MLX_PROJECT = Path(__file__).parent.parent
PYTORCH_PROJECT = Path.home() / "Projects/index-tts"
REF_AUDIO = MLX_PROJECT / "ref_audios/voice_01.wav"
OUTPUT_DIR = MLX_PROJECT


@dataclass
class TestResult:
    """Test result statistics."""
    name: str
    output_file: str
    success: bool = False
    audio_duration: float = 0.0
    total_time: float = 0.0
    rtf: float = 0.0
    gpt_time: float = 0.0
    s2mel_time: float = 0.0
    bigvgan_time: float = 0.0
    error: Optional[str] = None


def parse_mlx_output(output: str, result: TestResult):
    """Parse MLX output for statistics."""
    # Audio duration: 15.98s
    m = re.search(r'Audio duration: ([\d.]+)s', output)
    if m:
        result.audio_duration = float(m.group(1))

    # Total time: 22.08s (RTF: 1.38)
    m = re.search(r'Total time: ([\d.]+)s \(RTF: ([\d.]+)\)', output)
    if m:
        result.total_time = float(m.group(1))
        result.rtf = float(m.group(2))

    # GPT gen: 10.94s
    m = re.search(r'GPT gen: ([\d.]+)s', output)
    if m:
        result.gpt_time = float(m.group(1))

    # S2Mel: 4.15s
    m = re.search(r'S2Mel: ([\d.]+)s', output)
    if m:
        result.s2mel_time = float(m.group(1))

    # BigVGAN: 5.60s
    m = re.search(r'BigVGAN: ([\d.]+)s', output)
    if m:
        result.bigvgan_time = float(m.group(1))

    # v1.5 format: Generated 18.47s audio in 10.32s (RTF: 0.558)
    m = re.search(r'Generated ([\d.]+)s audio in ([\d.]+)s \(RTF: ([\d.]+)\)', output)
    if m:
        result.audio_duration = float(m.group(1))
        result.total_time = float(m.group(2))
        result.rtf = float(m.group(3))


def parse_pytorch_output(output: str, result: TestResult):
    """Parse PyTorch output for statistics."""
    # >> Generated audio length: 16.47 seconds
    m = re.search(r'Generated audio length: ([\d.]+) seconds', output)
    if m:
        result.audio_duration = float(m.group(1))

    # >> Total inference time: 73.63 seconds
    m = re.search(r'Total inference time: ([\d.]+) seconds', output)
    if m:
        result.total_time = float(m.group(1))

    # >> RTF: 4.4708
    m = re.search(r'RTF: ([\d.]+)', output)
    if m:
        result.rtf = float(m.group(1))

    # >> gpt_gen_time: 69.94 seconds
    m = re.search(r'gpt_gen_time: ([\d.]+) seconds', output)
    if m:
        result.gpt_time = float(m.group(1))

    # >> s2mel_time: 8.44 seconds
    m = re.search(r's2mel_time: ([\d.]+) seconds', output)
    if m:
        result.s2mel_time = float(m.group(1))

    # >> bigvgan_time: 2.21 seconds
    m = re.search(r'bigvgan_time: ([\d.]+) seconds', output)
    if m:
        result.bigvgan_time = float(m.group(1))


def run_mlx_v15(text: str, output: str) -> TestResult:
    """Generate using MLX v1.5."""
    result = TestResult(name="MLX v1.5", output_file=output)
    cmd = [
        "uv", "run", "mlx-indextts", "generate",
        "-m", str(MLX_PROJECT / "models/mlx-indexTTS-1.5"),
        "-r", str(REF_AUDIO),
        "-t", text,
        "-o", output,
        "-v"
    ]
    print(f"\n{'='*60}")
    print(f"MLX v1.5 -> {output}")
    print(f"{'='*60}")

    try:
        proc = subprocess.run(cmd, cwd=MLX_PROJECT, capture_output=True, text=True)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        if proc.returncode == 0:
            result.success = True
            parse_mlx_output(proc.stdout + proc.stderr, result)
        else:
            result.error = proc.stderr or "Unknown error"
    except Exception as e:
        result.error = str(e)

    return result


def run_mlx_v20(text: str, output: str) -> TestResult:
    """Generate using MLX v2.0."""
    result = TestResult(name="MLX v2.0", output_file=output)
    cmd = [
        "uv", "run", "mlx-indextts", "generate",
        "-m", str(MLX_PROJECT / "models/mlx-indexTTS-2.0"),
        "-r", str(REF_AUDIO),
        "-t", text,
        "-o", output,
        "-v"
    ]
    print(f"\n{'='*60}")
    print(f"MLX v2.0 -> {output}")
    print(f"{'='*60}")

    try:
        proc = subprocess.run(cmd, cwd=MLX_PROJECT, capture_output=True, text=True)
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        if proc.returncode == 0:
            result.success = True
            parse_mlx_output(proc.stdout + proc.stderr, result)
        else:
            result.error = proc.stderr or "Unknown error"
    except Exception as e:
        result.error = str(e)

    return result


def run_pytorch_v15(text: str, output: str) -> TestResult:
    """Generate using PyTorch v1.5."""
    result = TestResult(name="PyTorch v1.5", output_file=output)
    model_dir = PYTORCH_PROJECT / "indexTTS-1.5"
    cfg_path = model_dir / "config.yaml"
    script = f'''
import sys
sys.path.insert(0, "{PYTORCH_PROJECT}")
from indextts.infer import IndexTTS

tts = IndexTTS(cfg_path="{cfg_path}", model_dir="{model_dir}", device="mps")
tts.infer("{REF_AUDIO}", "{text}", "{output}")
print(f"Saved: {output}")
'''
    print(f"\n{'='*60}")
    print(f"PyTorch v1.5 -> {output}")
    print(f"{'='*60}")

    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", script],
            cwd=PYTORCH_PROJECT,
            capture_output=True,
            text=True
        )
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        if proc.returncode == 0:
            result.success = True
            parse_pytorch_output(proc.stdout + proc.stderr, result)
        else:
            result.error = proc.stderr or "Unknown error"
    except Exception as e:
        result.error = str(e)

    return result


def run_pytorch_v20(text: str, output: str) -> TestResult:
    """Generate using PyTorch v2.0."""
    result = TestResult(name="PyTorch v2.0", output_file=output)
    model_dir = PYTORCH_PROJECT / "indexTTS-2"
    cfg_path = model_dir / "config.yaml"
    script = f'''
import sys
sys.path.insert(0, "{PYTORCH_PROJECT}")
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(cfg_path="{cfg_path}", model_dir="{model_dir}", device="mps")
tts.infer("{REF_AUDIO}", "{text}", "{output}")
print(f"Saved: {output}")
'''
    print(f"\n{'='*60}")
    print(f"PyTorch v2.0 -> {output}")
    print(f"{'='*60}")

    try:
        proc = subprocess.run(
            ["uv", "run", "python", "-c", script],
            cwd=PYTORCH_PROJECT,
            capture_output=True,
            text=True
        )
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)

        if proc.returncode == 0:
            result.success = True
            parse_pytorch_output(proc.stdout + proc.stderr, result)
        else:
            result.error = proc.stderr or "Unknown error"
    except Exception as e:
        result.error = str(e)

    return result


def print_summary(results: List[TestResult]):
    """Print summary table of all results."""
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Header
    print(f"{'Name':<15} {'Status':<8} {'Audio':<8} {'Total':<10} {'RTF':<8} {'GPT':<10} {'S2Mel':<10} {'BigVGAN':<10}")
    print("-" * 80)

    for r in results:
        status = "✅" if r.success else "❌"
        audio = f"{r.audio_duration:.1f}s" if r.audio_duration else "-"
        total = f"{r.total_time:.1f}s" if r.total_time else "-"
        rtf = f"{r.rtf:.2f}" if r.rtf else "-"
        gpt = f"{r.gpt_time:.1f}s" if r.gpt_time else "-"
        s2mel = f"{r.s2mel_time:.1f}s" if r.s2mel_time else "-"
        bigvgan = f"{r.bigvgan_time:.1f}s" if r.bigvgan_time else "-"

        print(f"{r.name:<15} {status:<8} {audio:<8} {total:<10} {rtf:<8} {gpt:<10} {s2mel:<10} {bigvgan:<10}")

    print("-" * 80)

    # File sizes
    print(f"\n{'Generated Files:'}")
    for r in results:
        if r.success and Path(r.output_file).exists():
            size = Path(r.output_file).stat().st_size / 1024
            print(f"  {Path(r.output_file).name}: {size:.0f}KB")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate test audio")
    parser.add_argument("--mlx-only", action="store_true", help="Only run MLX versions")
    parser.add_argument("--pytorch-only", action="store_true", help="Only run PyTorch versions")
    parser.add_argument("--v15-only", action="store_true", help="Only run v1.5")
    parser.add_argument("--v20-only", action="store_true", help="Only run v2.0")
    parser.add_argument("--chinese-only", action="store_true", help="Only Chinese")
    parser.add_argument("--english-only", action="store_true", help="Only English")
    args = parser.parse_args()

    # Determine what to run
    run_mlx = not args.pytorch_only
    run_pytorch = not args.mlx_only
    run_v15 = not args.v20_only
    run_v20 = not args.v15_only
    run_chinese = not args.english_only
    run_english = not args.chinese_only

    print("Test Audio Generation")
    print(f"  Reference: {REF_AUDIO}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  MLX: {run_mlx}, PyTorch: {run_pytorch}")
    print(f"  v1.5: {run_v15}, v2.0: {run_v20}")
    print(f"  Chinese: {run_chinese}, English: {run_english}")

    results: List[TestResult] = []

    # MLX v1.5
    if run_mlx and run_v15:
        if run_chinese:
            results.append(run_mlx_v15(CHINESE_TEXT, str(OUTPUT_DIR / "test_mlx_v15_chinese.wav")))
        if run_english:
            results.append(run_mlx_v15(ENGLISH_TEXT, str(OUTPUT_DIR / "test_mlx_v15_english.wav")))

    # MLX v2.0
    if run_mlx and run_v20:
        if run_chinese:
            results.append(run_mlx_v20(CHINESE_TEXT, str(OUTPUT_DIR / "test_mlx_v20_chinese.wav")))
        if run_english:
            results.append(run_mlx_v20(ENGLISH_TEXT, str(OUTPUT_DIR / "test_mlx_v20_english.wav")))

    # PyTorch v1.5
    if run_pytorch and run_v15:
        if run_chinese:
            results.append(run_pytorch_v15(CHINESE_TEXT, str(OUTPUT_DIR / "test_pytorch_v15_chinese.wav")))
        if run_english:
            results.append(run_pytorch_v15(ENGLISH_TEXT, str(OUTPUT_DIR / "test_pytorch_v15_english.wav")))

    # PyTorch v2.0
    if run_pytorch and run_v20:
        if run_chinese:
            results.append(run_pytorch_v20(CHINESE_TEXT, str(OUTPUT_DIR / "test_pytorch_v20_chinese.wav")))
        if run_english:
            results.append(run_pytorch_v20(ENGLISH_TEXT, str(OUTPUT_DIR / "test_pytorch_v20_english.wav")))

    # Print summary
    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
