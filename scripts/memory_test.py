#!/usr/bin/env python3
"""Memory usage comparison between PyTorch and MLX implementations."""

import os
import sys
import time
import subprocess
import argparse


def get_memory_usage():
    """Get current process memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_pytorch_memory():
    """Test PyTorch IndexTTS memory usage."""
    print("=" * 60)
    print("Testing PyTorch IndexTTS Memory Usage")
    print("=" * 60)

    # Add PyTorch IndexTTS to path
    sys.path.insert(0, os.path.expanduser("~/Projects/index-tts"))

    import torch
    import gc

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mem_start = get_memory_usage()
    print(f"Initial memory: {mem_start:.1f} MB")

    # Load model
    from indextts.infer import IndexTTS

    model_dir = os.path.expanduser("~/Projects/index-tts/checkpoints/indexTTS-1.5")

    print(f"\nLoading model from {model_dir}...")
    tts = IndexTTS(model_dir=model_dir, cfg_path=os.path.join(model_dir, "config.yaml"))

    mem_after_load = get_memory_usage()
    print(f"Memory after model load: {mem_after_load:.1f} MB (+{mem_after_load - mem_start:.1f} MB)")

    # Run inference
    ref_audio = os.path.expanduser("~/Projects/mlx-indextts/ref_audios/voice_01.wav")
    text = "今天天气真不错，我们一起出去散步吧。"

    print(f"\nGenerating speech...")
    print(f"  Reference: {ref_audio}")
    print(f"  Text: {text}")

    start_time = time.perf_counter()

    # Track peak memory during generation
    import threading
    peak_mem = [mem_after_load]
    stop_monitoring = threading.Event()

    def monitor_memory():
        while not stop_monitoring.is_set():
            current = get_memory_usage()
            if current > peak_mem[0]:
                peak_mem[0] = current
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    # Generate
    sr, audio = tts.infer(ref_audio, text)

    stop_monitoring.set()
    monitor_thread.join()

    elapsed = time.perf_counter() - start_time
    mem_after_gen = get_memory_usage()

    print(f"\nResults:")
    print(f"  Generation time: {elapsed:.2f}s")
    print(f"  Audio duration: {len(audio) / sr:.2f}s")
    print(f"  RTF: {elapsed / (len(audio) / sr):.3f}")
    print(f"  Memory after generation: {mem_after_gen:.1f} MB")
    print(f"  Peak memory: {peak_mem[0]:.1f} MB")
    print(f"  Memory increase during generation: +{peak_mem[0] - mem_after_load:.1f} MB")

    return {
        "initial": mem_start,
        "after_load": mem_after_load,
        "after_gen": mem_after_gen,
        "peak": peak_mem[0],
        "time": elapsed,
        "audio_duration": len(audio) / sr,
    }


def test_mlx_memory():
    """Test MLX IndexTTS memory usage."""
    print("=" * 60)
    print("Testing MLX IndexTTS Memory Usage")
    print("=" * 60)

    import mlx.core as mx
    import gc

    # Force garbage collection
    gc.collect()

    mem_start = get_memory_usage()
    print(f"Initial memory: {mem_start:.1f} MB")

    # Load model
    sys.path.insert(0, os.path.expanduser("~/Projects/mlx-indextts"))
    from mlx_indextts.generate import IndexTTS

    model_dir = os.path.expanduser("~/Projects/mlx-indextts/models/mlx-indexTTS-1.5")

    print(f"\nLoading model from {model_dir}...")
    tts = IndexTTS.load_model(model_dir)
    mx.eval(tts.gpt.parameters(), tts.bigvgan.parameters())

    mem_after_load = get_memory_usage()
    print(f"Memory after model load: {mem_after_load:.1f} MB (+{mem_after_load - mem_start:.1f} MB)")

    # Run inference
    ref_audio = os.path.expanduser("~/Projects/mlx-indextts/ref_audios/voice_01.wav")
    text = "今天天气真不错，我们一起出去散步吧。"

    print(f"\nGenerating speech...")
    print(f"  Reference: {ref_audio}")
    print(f"  Text: {text}")

    start_time = time.perf_counter()

    # Track peak memory during generation
    import threading
    peak_mem = [mem_after_load]
    stop_monitoring = threading.Event()

    def monitor_memory():
        while not stop_monitoring.is_set():
            current = get_memory_usage()
            if current > peak_mem[0]:
                peak_mem[0] = current
            time.sleep(0.1)

    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    # Generate
    audio = tts.generate(text, ref_audio, verbose=True)
    mx.eval(audio)

    stop_monitoring.set()
    monitor_thread.join()

    elapsed = time.perf_counter() - start_time
    mem_after_gen = get_memory_usage()
    audio_duration = len(audio) / tts.sample_rate

    print(f"\nResults:")
    print(f"  Generation time: {elapsed:.2f}s")
    print(f"  Audio duration: {audio_duration:.2f}s")
    print(f"  RTF: {elapsed / audio_duration:.3f}")
    print(f"  Memory after generation: {mem_after_gen:.1f} MB")
    print(f"  Peak memory: {peak_mem[0]:.1f} MB")
    print(f"  Memory increase during generation: +{peak_mem[0] - mem_after_load:.1f} MB")

    return {
        "initial": mem_start,
        "after_load": mem_after_load,
        "after_gen": mem_after_gen,
        "peak": peak_mem[0],
        "time": elapsed,
        "audio_duration": audio_duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Memory usage comparison")
    parser.add_argument("--pytorch", action="store_true", help="Test PyTorch only")
    parser.add_argument("--mlx", action="store_true", help="Test MLX only")
    args = parser.parse_args()

    results = {}

    if args.pytorch or (not args.pytorch and not args.mlx):
        try:
            results["pytorch"] = test_pytorch_memory()
        except Exception as e:
            print(f"PyTorch test failed: {e}")
            import traceback
            traceback.print_exc()

    if args.mlx or (not args.pytorch and not args.mlx):
        try:
            results["mlx"] = test_mlx_memory()
        except Exception as e:
            print(f"MLX test failed: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"{'Metric':<30} {'PyTorch':>12} {'MLX':>12} {'Diff':>12}")
        print("-" * 66)

        pt = results["pytorch"]
        mlx = results["mlx"]

        print(f"{'Model load memory (MB)':<30} {pt['after_load'] - pt['initial']:>12.1f} {mlx['after_load'] - mlx['initial']:>12.1f} {(mlx['after_load'] - mlx['initial']) - (pt['after_load'] - pt['initial']):>+12.1f}")
        print(f"{'Peak memory (MB)':<30} {pt['peak']:>12.1f} {mlx['peak']:>12.1f} {mlx['peak'] - pt['peak']:>+12.1f}")
        print(f"{'Generation time (s)':<30} {pt['time']:>12.2f} {mlx['time']:>12.2f} {mlx['time'] - pt['time']:>+12.2f}")
        print(f"{'RTF':<30} {pt['time']/pt['audio_duration']:>12.3f} {mlx['time']/mlx['audio_duration']:>12.3f}")


if __name__ == "__main__":
    main()
