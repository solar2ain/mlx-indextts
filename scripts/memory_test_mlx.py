#!/usr/bin/env python3
"""MLX IndexTTS memory test with Metal memory tracking."""

import os
import sys
import time
import subprocess

def get_memory_footprint():
    """Get memory footprint using macOS footprint command."""
    pid = os.getpid()
    try:
        result = subprocess.run(
            ["footprint", "-p", str(pid), "--skip-aggregation"],
            capture_output=True, text=True
        )
        # Parse the output for total footprint
        for line in result.stdout.split('\n'):
            if 'Total:' in line or 'TOTAL' in line:
                # Extract MB value
                import re
                match = re.search(r'([\d.]+)\s*[MG]B', line)
                if match:
                    val = float(match.group(1))
                    if 'GB' in line:
                        val *= 1024
                    return val
    except:
        pass
    return None

def get_process_memory():
    """Get process memory in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    return {
        "rss": info.rss / 1024 / 1024,
        "vms": info.vms / 1024 / 1024,
    }

def main():
    print("=" * 60)
    print("MLX IndexTTS Memory Usage Test")
    print("=" * 60)

    import mlx.core as mx
    import gc

    gc.collect()

    mem_start = get_process_memory()
    print(f"Initial memory: RSS={mem_start['rss']:.1f} MB")

    # Load model
    sys.path.insert(0, os.path.expanduser("~/Projects/mlx-indextts"))
    from mlx_indextts.generate import IndexTTS

    model_dir = os.path.expanduser("~/Projects/mlx-indextts/models/mlx-indexTTS-1.5")

    print(f"\nLoading model from {model_dir}...")
    tts = IndexTTS.load_model(model_dir)

    # Force evaluation of model parameters
    mx.eval(tts.gpt.parameters(), tts.bigvgan.parameters())

    mem_after_load = get_process_memory()
    print(f"Memory after model load: RSS={mem_after_load['rss']:.1f} MB (+{mem_after_load['rss'] - mem_start['rss']:.1f} MB)")

    # Check MLX memory
    print(f"\nMLX active memory: {mx.metal.get_active_memory() / 1024 / 1024:.1f} MB")
    print(f"MLX peak memory: {mx.metal.get_peak_memory() / 1024 / 1024:.1f} MB")
    print(f"MLX cache memory: {mx.metal.get_cache_memory() / 1024 / 1024:.1f} MB")

    # Run inference
    ref_audio = os.path.expanduser("~/Projects/mlx-indextts/ref_audios/voice_01.wav")
    text = "今天天气真不错，我们一起出去散步吧。这是一个测试语音合成的长句子。"

    print(f"\nGenerating speech...")
    print(f"  Reference: {ref_audio}")
    print(f"  Text: {text}")

    # Reset peak memory counter
    mx.metal.reset_peak_memory()

    import threading
    peak_rss = [mem_after_load['rss']]
    stop_monitoring = threading.Event()

    def monitor_memory():
        while not stop_monitoring.is_set():
            current = get_process_memory()
            if current['rss'] > peak_rss[0]:
                peak_rss[0] = current['rss']
            time.sleep(0.05)

    monitor_thread = threading.Thread(target=monitor_memory)
    monitor_thread.start()

    start_time = time.perf_counter()
    audio = tts.generate(text, ref_audio, verbose=True)
    mx.eval(audio)
    elapsed = time.perf_counter() - start_time

    stop_monitoring.set()
    monitor_thread.join()

    mem_after_gen = get_process_memory()
    audio_duration = len(audio) / tts.sample_rate

    print(f"\n{'='*60}")
    print("Results Summary")
    print("=" * 60)
    print(f"Generation time: {elapsed:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"RTF: {elapsed / audio_duration:.3f}")
    print(f"\nMemory (RSS):")
    print(f"  Initial: {mem_start['rss']:.1f} MB")
    print(f"  After load: {mem_after_load['rss']:.1f} MB (+{mem_after_load['rss'] - mem_start['rss']:.1f} MB)")
    print(f"  After generation: {mem_after_gen['rss']:.1f} MB")
    print(f"  Peak RSS: {peak_rss[0]:.1f} MB")
    print(f"\nMLX Metal Memory:")
    print(f"  Active: {mx.metal.get_active_memory() / 1024 / 1024:.1f} MB")
    print(f"  Peak: {mx.metal.get_peak_memory() / 1024 / 1024:.1f} MB")
    print(f"  Cache: {mx.metal.get_cache_memory() / 1024 / 1024:.1f} MB")

    # Second generation to see memory behavior
    print(f"\n{'='*60}")
    print("Second Generation (warm run)")
    print("=" * 60)

    mx.metal.reset_peak_memory()
    gc.collect()

    start_time = time.perf_counter()
    audio2 = tts.generate("这是第二次生成测试。", ref_audio, verbose=False)
    mx.eval(audio2)
    elapsed2 = time.perf_counter() - start_time

    print(f"Generation time: {elapsed2:.2f}s")
    print(f"MLX Peak memory: {mx.metal.get_peak_memory() / 1024 / 1024:.1f} MB")
    print(f"MLX Active memory: {mx.metal.get_active_memory() / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
