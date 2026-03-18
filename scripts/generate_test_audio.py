#!/usr/bin/env python3
"""Generate test audio using both MLX and PyTorch versions.

Usage:
    uv run python scripts/generate_test_audio.py                    # Default: Chinese + English test
    uv run python scripts/generate_test_audio.py --quantize-test    # Quantization benchmark (fp32, 8-bit, 4-bit)
    uv run python scripts/generate_test_audio.py --emotion-test     # v2.0 emotion test (happy, sad)
    uv run python scripts/generate_test_audio.py -t "Hello" -r voice.wav  # Custom text/audio
    uv run python scripts/generate_test_audio.py --v20-only --mlx-only    # Only MLX v2.0

Output files will be saved to test_outputs/ directory.
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
SHORT_TEXT = "这是一个简短的测试文本，用于快速验证量化效果。"

# Paths
MLX_PROJECT = Path(__file__).parent.parent
PYTORCH_PROJECT = Path.home() / "Projects/index-tts"
REF_AUDIO = MLX_PROJECT / "ref_audios/voice_01.wav"
OUTPUT_DIR = MLX_PROJECT / "test_outputs"
SPEAKER_V20_CACHE = OUTPUT_DIR / "speaker_v20_cache.npz"


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
    peak_memory_mb: float = 0.0
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

    # v1.5 GPT gen: 8.50s
    m = re.search(r'GPT gen: ([\d.]+)s', output)
    if m:
        result.gpt_time = float(m.group(1))

    # v1.5 BigVGAN: 1.50s
    m = re.search(r'BigVGAN: ([\d.]+)s', output)
    if m:
        result.bigvgan_time = float(m.group(1))

    # Peak memory: 1234.5 MB
    m = re.search(r'Peak memory: ([\d.]+)\s*MB', output)
    if m:
        result.peak_memory_mb = float(m.group(1))


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


def run_mlx_v15(text: str, output: str, quantize: Optional[str] = None) -> TestResult:
    """Generate using MLX v1.5."""
    name = f"MLX v1.5" + (f" q{quantize}" if quantize else "")
    result = TestResult(name=name, output_file=output)
    cmd = [
        "uv", "run", "mlx-indextts", "generate",
        "-m", str(MLX_PROJECT / "models/mlx-indexTTS-1.5"),
        "-r", str(REF_AUDIO),
        "-t", text,
        "-o", output,
        "-v"
    ]
    if quantize:
        cmd.extend(["--quantize", quantize])

    print(f"\n{'='*60}")
    print(f"{name} -> {output}")
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


def run_mlx_v20(text: str, output: str, quantize: Optional[str] = None, emotion: Optional[str] = None, emo_alpha: float = 1.0, use_speaker_cache: bool = True) -> TestResult:
    """Generate using MLX v2.0."""
    name = f"MLX v2.0" + (f" q{quantize}" if quantize else "") + (f" {emotion}" if emotion else "")
    result = TestResult(name=name, output_file=output)

    # Use speaker cache for faster testing
    ref_audio = REF_AUDIO
    if use_speaker_cache:
        ref_audio = ensure_v20_speaker_cache(REF_AUDIO, SPEAKER_V20_CACHE)

    cmd = [
        "uv", "run", "mlx-indextts", "generate",
        "-m", str(MLX_PROJECT / "models/mlx-indexTTS-2.0"),
        "-r", str(ref_audio),
        "-t", text,
        "-o", output,
        "-v"
    ]
    if quantize:
        cmd.extend(["--quantize", quantize])
    if emotion:
        cmd.extend(["--emotion", emotion, "--emo-alpha", str(emo_alpha)])

    print(f"\n{'='*60}")
    print(f"{name} -> {output}")
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


def run_pytorch_v20(text: str, output: str, emotion: Optional[str] = None, emo_alpha: float = 1.0) -> TestResult:
    """Generate using PyTorch v2.0."""
    name = "PyTorch v2.0" + (f" {emotion}" if emotion else "")
    result = TestResult(name=name, output_file=output)
    model_dir = PYTORCH_PROJECT / "indexTTS-2"
    cfg_path = model_dir / "config.yaml"

    # Build emotion args
    emotion_args = ""
    if emotion:
        emotion_args = f', emotion="{emotion}", emo_alpha={emo_alpha}'

    script = f'''
import sys
sys.path.insert(0, "{PYTORCH_PROJECT}")
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(cfg_path="{cfg_path}", model_dir="{model_dir}", device="mps")
tts.infer("{REF_AUDIO}", "{text}", "{output}"{emotion_args})
print(f"Saved: {output}")
'''
    print(f"\n{'='*60}")
    print(f"{name} -> {output}")
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


def get_metal_memory() -> float:
    """Get current Metal memory usage in MB."""
    try:
        import mlx.core as mx
        # Force sync to get accurate memory
        mx.eval(mx.zeros(1))
        peak = mx.metal.get_peak_memory() / 1024 / 1024
        return peak
    except:
        return 0.0


def ensure_v20_speaker_cache(ref_audio: Path, cache_path: Path) -> Path:
    """Pre-compute v2.0 speaker cache if not exists or outdated.

    Returns the cache path to use as ref_audio.
    """
    # Check if cache exists and is newer than ref_audio
    if cache_path.exists():
        if cache_path.stat().st_mtime >= ref_audio.stat().st_mtime:
            print(f"Using cached speaker: {cache_path}")
            return cache_path
        else:
            print(f"Speaker cache outdated, regenerating...")
    else:
        print(f"Pre-computing v2.0 speaker conditioning...")

    # Generate speaker cache
    cmd = [
        "uv", "run", "mlx-indextts", "speaker",
        "-m", str(MLX_PROJECT / "models/mlx-indexTTS-2.0"),
        "-r", str(ref_audio),
        "-o", str(cache_path),
    ]

    try:
        proc = subprocess.run(cmd, cwd=MLX_PROJECT, capture_output=True, text=True)
        if proc.returncode == 0:
            print(f"Speaker cache saved: {cache_path}")
            return cache_path
        else:
            print(f"Failed to create speaker cache: {proc.stderr}")
            return ref_audio  # Fallback to original audio
    except Exception as e:
        print(f"Error creating speaker cache: {e}")
        return ref_audio


def run_quantize_benchmark(run_v15: bool = True, run_v20: bool = True, text: Optional[str] = None, use_speaker_cache: bool = True):
    """Run quantization benchmark comparing fp32, 8-bit, and 4-bit."""
    test_text = text if text else CHINESE_TEXT

    print("\n" + "="*80)
    print("QUANTIZATION BENCHMARK")
    print("="*80)
    print(f"Text: {test_text[:50]}...")
    print(f"Reference: {REF_AUDIO}")

    results = []

    # Test v1.5 with different quantization levels
    if run_v15:
        print("\n--- v1.5 Quantization Test ---")
        for q in [None, "8", "4"]:
            q_label = q if q else "fp32"
            output = str(OUTPUT_DIR / f"test_v15_q{q_label}.wav")
            result = run_mlx_v15(test_text, output, quantize=q)
            results.append(result)

    # Test v2.0 with different quantization levels
    if run_v20:
        print("\n--- v2.0 Quantization Test ---")
        for q in [None, "8", "4"]:
            q_label = q if q else "fp32"
            output = str(OUTPUT_DIR / f"test_v20_q{q_label}.wav")
            result = run_mlx_v20(test_text, output, quantize=q, use_speaker_cache=use_speaker_cache)
            results.append(result)

    # Print quantization summary
    print_quantize_summary(results)

    return results


def run_emotion_benchmark(text: Optional[str] = None, run_pytorch: bool = True, use_speaker_cache: bool = True):
    """Run emotion benchmark for v2.0 (MLX and PyTorch)."""
    test_text = text if text else "今天真是太开心了！我收到了期待已久的礼物，心情特别好！"

    print("\n" + "="*80)
    print("EMOTION BENCHMARK (v2.0)")
    print("="*80)
    print(f"Text: {test_text[:50]}...")
    print(f"Reference: {REF_AUDIO}")

    results = []

    # MLX tests
    print("\n--- MLX Baseline (no emotion) ---")
    output = str(OUTPUT_DIR / "test_mlx_v20_no_emotion.wav")
    result = run_mlx_v20(test_text, output, use_speaker_cache=use_speaker_cache)
    results.append(result)

    for emotion in ["happy", "sad"]:
        print(f"\n--- MLX Emotion: {emotion} ---")
        output = str(OUTPUT_DIR / f"test_mlx_v20_emotion_{emotion}.wav")
        result = run_mlx_v20(test_text, output, emotion=emotion, emo_alpha=0.8, use_speaker_cache=use_speaker_cache)
        results.append(result)

    # PyTorch tests
    if run_pytorch:
        print("\n--- PyTorch Baseline (no emotion) ---")
        output = str(OUTPUT_DIR / "test_pytorch_v20_no_emotion.wav")
        result = run_pytorch_v20(test_text, output)
        results.append(result)

        for emotion in ["happy", "sad"]:
            print(f"\n--- PyTorch Emotion: {emotion} ---")
            output = str(OUTPUT_DIR / f"test_pytorch_v20_emotion_{emotion}.wav")
            result = run_pytorch_v20(test_text, output, emotion=emotion, emo_alpha=0.8)
            results.append(result)

    # Print summary
    print_summary(results)

    return results


def print_quantize_summary(results: List[TestResult]):
    """Print quantization comparison summary."""
    print(f"\n{'='*90}")
    print("QUANTIZATION COMPARISON")
    print(f"{'='*90}")

    # Header
    print(f"{'Name':<18} {'Status':<8} {'Audio':<8} {'Total':<10} {'RTF':<8} {'GPT':<10} {'Memory':<12}")
    print("-" * 90)

    for r in results:
        status = "✅" if r.success else "❌"
        audio = f"{r.audio_duration:.1f}s" if r.audio_duration else "-"
        total = f"{r.total_time:.2f}s" if r.total_time else "-"
        rtf = f"{r.rtf:.3f}" if r.rtf else "-"
        gpt = f"{r.gpt_time:.2f}s" if r.gpt_time else "-"
        memory = f"{r.peak_memory_mb:.0f}MB" if r.peak_memory_mb else "-"

        print(f"{r.name:<18} {status:<8} {audio:<8} {total:<10} {rtf:<8} {gpt:<10} {memory:<12}")

    print("-" * 90)

    # Calculate speedup
    v15_results = [r for r in results if "v1.5" in r.name and r.success]
    v20_results = [r for r in results if "v2.0" in r.name and r.success]

    for version_results, version in [(v15_results, "v1.5"), (v20_results, "v2.0")]:
        if len(version_results) >= 2:
            fp32 = next((r for r in version_results if "fp32" in r.name or "q" not in r.name.split()[-1]), None)
            q8 = next((r for r in version_results if "q8" in r.name), None)
            q4 = next((r for r in version_results if "q4" in r.name), None)

            print(f"\n{version} Speedup:")
            if fp32 and q8 and fp32.total_time > 0:
                speedup = fp32.total_time / q8.total_time
                print(f"  8-bit vs fp32: {speedup:.2f}x")
            if fp32 and q4 and fp32.total_time > 0:
                speedup = fp32.total_time / q4.total_time
                print(f"  4-bit vs fp32: {speedup:.2f}x")


def print_summary(results: List[TestResult]):
    """Print summary table of all results."""
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    # Header
    print(f"{'Name':<18} {'Status':<8} {'Audio':<8} {'Total':<10} {'RTF':<8} {'GPT':<10} {'S2Mel':<10} {'BigVGAN':<10}")
    print("-" * 90)

    for r in results:
        status = "✅" if r.success else "❌"
        audio = f"{r.audio_duration:.1f}s" if r.audio_duration else "-"
        total = f"{r.total_time:.1f}s" if r.total_time else "-"
        rtf = f"{r.rtf:.2f}" if r.rtf else "-"
        gpt = f"{r.gpt_time:.1f}s" if r.gpt_time else "-"
        s2mel = f"{r.s2mel_time:.1f}s" if r.s2mel_time else "-"
        bigvgan = f"{r.bigvgan_time:.1f}s" if r.bigvgan_time else "-"

        print(f"{r.name:<18} {status:<8} {audio:<8} {total:<10} {rtf:<8} {gpt:<10} {s2mel:<10} {bigvgan:<10}")

    print("-" * 90)

    # File sizes
    print(f"\n{'Generated Files:'}")
    for r in results:
        if r.success and Path(r.output_file).exists():
            size = Path(r.output_file).stat().st_size / 1024
            print(f"  {Path(r.output_file).name}: {size:.0f}KB")


def main():
    global REF_AUDIO, OUTPUT_DIR, SPEAKER_V20_CACHE

    import argparse
    parser = argparse.ArgumentParser(description="Generate test audio")
    parser.add_argument("--mlx-only", action="store_true", help="Only run MLX versions")
    parser.add_argument("--pytorch-only", action="store_true", help="Only run PyTorch versions")
    parser.add_argument("--v15-only", action="store_true", help="Only run v1.5")
    parser.add_argument("--v20-only", action="store_true", help="Only run v2.0")
    parser.add_argument("--chinese-only", action="store_true", help="Only Chinese")
    parser.add_argument("--english-only", action="store_true", help="Only English")
    parser.add_argument("--quantize-test", action="store_true", help="Run quantization benchmark (fp32, 8-bit, 4-bit)")
    parser.add_argument("-q", "--quantize", type=str, default=None, help="Quantization level (4, 8, or fp32)")
    parser.add_argument("-t", "--text", type=str, default=None, help="Custom text to synthesize")
    parser.add_argument("-r", "--ref-audio", type=str, default=None, help="Custom reference audio file")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory (default: test_outputs/)")
    parser.add_argument("--emotion-test", action="store_true", help="Run v2.0 emotion test (happy, sad)")
    parser.add_argument("--no-speaker-cache", action="store_true", help="Disable v2.0 speaker cache (use raw audio)")
    args = parser.parse_args()

    # Update paths if specified
    if args.ref_audio:
        REF_AUDIO = Path(args.ref_audio).expanduser().resolve()
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
        SPEAKER_V20_CACHE = OUTPUT_DIR / "speaker_v20_cache.npz"

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run emotion benchmark if requested
    if args.emotion_test:
        run_emotion_benchmark(text=args.text, run_pytorch=not args.mlx_only, use_speaker_cache=not args.no_speaker_cache)
        return

    # Run quantization benchmark if requested
    if args.quantize_test:
        run_quantize_benchmark(
            run_v15=not args.v20_only,
            run_v20=not args.v15_only,
            text=args.text,
            use_speaker_cache=not args.no_speaker_cache,
        )
        return

    # Determine what to run
    run_mlx = not args.pytorch_only
    run_pytorch = not args.mlx_only
    run_v15 = not args.v20_only
    run_v20 = not args.v15_only
    run_chinese = not args.english_only
    run_english = not args.chinese_only

    # Use custom text if provided
    chinese_text = args.text if args.text else CHINESE_TEXT
    english_text = args.text if args.text else ENGLISH_TEXT

    print("Test Audio Generation")
    print(f"  Reference: {REF_AUDIO}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  MLX: {run_mlx}, PyTorch: {run_pytorch}")
    print(f"  v1.5: {run_v15}, v2.0: {run_v20}")
    if args.text:
        print(f"  Text: {args.text[:50]}...")
    else:
        print(f"  Chinese: {run_chinese}, English: {run_english}")
    if args.quantize:
        print(f"  Quantize: {args.quantize}")

    results: List[TestResult] = []
    use_speaker_cache = not args.no_speaker_cache

    # If custom text provided, use unified output names
    if args.text:
        if run_mlx and run_v15:
            results.append(run_mlx_v15(args.text, str(OUTPUT_DIR / "test_mlx_v15.wav"), args.quantize))
        if run_mlx and run_v20:
            results.append(run_mlx_v20(args.text, str(OUTPUT_DIR / "test_mlx_v20.wav"), args.quantize, use_speaker_cache=use_speaker_cache))
        if run_pytorch and run_v15:
            results.append(run_pytorch_v15(args.text, str(OUTPUT_DIR / "test_pytorch_v15.wav")))
        if run_pytorch and run_v20:
            results.append(run_pytorch_v20(args.text, str(OUTPUT_DIR / "test_pytorch_v20.wav")))
    else:
        # MLX v1.5
        if run_mlx and run_v15:
            if run_chinese:
                results.append(run_mlx_v15(chinese_text, str(OUTPUT_DIR / "test_mlx_v15_chinese.wav"), args.quantize))
            if run_english:
                results.append(run_mlx_v15(english_text, str(OUTPUT_DIR / "test_mlx_v15_english.wav"), args.quantize))

        # MLX v2.0
        if run_mlx and run_v20:
            if run_chinese:
                results.append(run_mlx_v20(chinese_text, str(OUTPUT_DIR / "test_mlx_v20_chinese.wav"), args.quantize, use_speaker_cache=use_speaker_cache))
            if run_english:
                results.append(run_mlx_v20(english_text, str(OUTPUT_DIR / "test_mlx_v20_english.wav"), args.quantize, use_speaker_cache=use_speaker_cache))

        # PyTorch v1.5
        if run_pytorch and run_v15:
            if run_chinese:
                results.append(run_pytorch_v15(chinese_text, str(OUTPUT_DIR / "test_pytorch_v15_chinese.wav")))
            if run_english:
                results.append(run_pytorch_v15(english_text, str(OUTPUT_DIR / "test_pytorch_v15_english.wav")))

        # PyTorch v2.0
        if run_pytorch and run_v20:
            if run_chinese:
                results.append(run_pytorch_v20(chinese_text, str(OUTPUT_DIR / "test_pytorch_v20_chinese.wav")))
            if run_english:
                results.append(run_pytorch_v20(english_text, str(OUTPUT_DIR / "test_pytorch_v20_english.wav")))

    # Print summary
    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
