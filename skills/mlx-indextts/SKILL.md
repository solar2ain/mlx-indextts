---
name: mlx-indextts
description: Generate speech from text using MLX-IndexTTS on Apple Silicon. Use this skill whenever the user wants to: synthesize speech, clone voices, generate audio from text, use TTS (text-to-speech), convert voice, or mentions IndexTTS/MLX-IndexTTS. Also trigger when users ask about voice cloning, speaker conditioning, or generating audio files from text input.
---

# MLX-IndexTTS

MLX-IndexTTS is an MLX implementation of IndexTTS that runs natively on Apple Silicon without PyTorch runtime dependency.

## Project Location

```
~/Projects/mlx-indextts
```

All commands must be executed in this directory using `uv run`.

## Quick Start

### Generate Speech

**Always use `--quantize 8` for better performance and lower memory usage.**

```bash
cd ~/Projects/mlx-indextts
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r <reference_audio.wav> \
    -t "Text to synthesize" \
    -o output.wav \
    --quantize 8 \
    --play  # Optional: play after generation
```

**Parameters:**
- `-m, --model`: MLX model directory path
- `-r, --ref-audio`: Reference audio file (.wav) or pre-computed speaker file (.npz)
- `-t, --text`: Text to synthesize
- `-o, --output`: Output audio file path
- `-q, --quantize`: Runtime quantization (4, 8, or fp32). **Default to 8 for best speed/quality balance**
- `--seed`: Random seed for reproducible generation
- `--play`: Auto-play after generation
- `--memory-limit`: GPU memory limit in GB (default: 8.0, 0 for no limit)
- `--max-tokens`: Maximum mel tokens (default: 600)
- `--temperature`: Sampling temperature (default: 1.0)
- `--top-k`: Top-k sampling (default: 30)
- `--top-p`: Top-p sampling (default: 0.8)

### Pre-compute Speaker (Faster Inference)

Pre-computing speaker conditioning reduces conditioning time from ~23ms to ~0.7ms (30x speedup):

```bash
cd ~/Projects/mlx-indextts
# Save speaker
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-1.5 \
    -r <reference_audio.wav> \
    -o <speaker.npz>

# Generate using pre-computed speaker
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r <speaker.npz> \
    -t "Hello, world!" \
    -o output.wav
```

### Convert Model

Convert PyTorch IndexTTS model to MLX format (requires torch):

```bash
cd ~/Projects/mlx-indextts
uv sync --extra convert  # Install torch dependency

uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    --output-dir models/mlx-indexTTS-1.5
```

## Available Models

| Model | Path | Status |
|-------|------|--------|
| IndexTTS 1.0 | `models/mlx-index-TTS` | ✅ |
| IndexTTS 1.5 | `models/mlx-indexTTS-1.5` | ✅ |
| IndexTTS 2.0 | - | Coming soon |

## Python API

```python
from mlx_indextts import IndexTTS

# Load model with 8-bit quantization (recommended)
tts = IndexTTS.load_model(
    "~/Projects/mlx-indextts/models/mlx-indexTTS-1.5",
    quantize_bits=8,  # Runtime quantization for better performance
)

# Generate speech
audio = tts.generate(
    text="Hello, this is a speech synthesis test.",
    ref_audio="./reference.wav",
    seed=42,  # Optional: for reproducibility
)

# Save audio
tts.save_audio(audio, "./output.wav")

# Pre-compute speaker
tts.save_speaker("./reference.wav", "./speaker.npz")
```

## Common Tasks

### 1. Clone someone's voice to say something
```bash
cd ~/Projects/mlx-indextts
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r person_recording.wav \
    -t "Content to speak" \
    -o output.wav \
    --quantize 8 --play
```

### 2. Batch generate multiple audio files
Pre-compute speaker first, then reuse:
```bash
cd ~/Projects/mlx-indextts
uv run mlx-indextts speaker -m models/mlx-indexTTS-1.5 -r voice.wav -o voice.npz
uv run mlx-indextts generate -m models/mlx-indexTTS-1.5 -r voice.npz -t "First sentence" -o out1.wav -q 8
uv run mlx-indextts generate -m models/mlx-indexTTS-1.5 -r voice.npz -t "Second sentence" -o out2.wav -q 8
```

### 3. Ensure reproducible generation
Use `--seed` parameter:
```bash
cd ~/Projects/mlx-indextts
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r voice.wav \
    -t "Hello" \
    -o output.wav \
    --quantize 8 --seed 42
```

## Performance

| Metric | Value |
|--------|-------|
| RTF (M3 Pro, 8-bit) | ~0.44 (2.3x faster than real-time) |
| RTF (M3 Pro, fp32) | ~0.52 (1.9x faster than real-time) |
| Conditioning (wav) | ~23ms |
| Conditioning (npz) | ~0.7ms |

### Runtime Quantization

Use `--quantize` for better performance without re-converting models:

| Quantization | Memory | Speed (RTF) | Quality |
|--------------|--------|-------------|---------|
| fp32 | Baseline | ~0.52 | Best |
| **8-bit (recommended)** | ~72% | ~0.44 | Good |
| 4-bit | ~60% | ~0.43 | Acceptable |

## Troubleshooting

### ModuleNotFoundError: No module named 'torch'
Model conversion requires torch:
```bash
cd ~/Projects/mlx-indextts
uv sync --extra convert
```

### Missing words in generated audio
Ensure you're using the latest version - mel position encoding bug has been fixed.

### Want faster inference?
1. Use `--quantize 8` for 8-bit quantization (15-20% faster)
2. Use pre-computed speaker (.npz) instead of raw audio (.wav)
