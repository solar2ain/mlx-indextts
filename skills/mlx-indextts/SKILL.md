---
name: mlx-indextts
description: Generate speech from text using MLX-IndexTTS on Apple Silicon. Use this skill whenever the user wants to: synthesize speech, clone voices, generate audio from text, use TTS (text-to-speech), convert voice, or mentions IndexTTS/MLX-IndexTTS. Also trigger when users ask about voice cloning, speaker conditioning, emotion control, or generating audio files from text input.
---

# MLX-IndexTTS

MLX-IndexTTS is an MLX implementation of IndexTTS that runs natively on Apple Silicon. Supports both v1.5 and v2.0 with automatic version detection.

## Project Location

```
~/Projects/mlx-indextts
```

All commands must be executed in this directory using `uv run`.

## Quick Start

### Generate Speech (Auto-detects version)

```bash
cd ~/Projects/mlx-indextts

# v1.5
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r <reference_audio.wav> \
    -t "Text to synthesize" \
    -o output.wav \
    --quantize 8 \
    --play  # Optional: play after generation

# v2.0
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r <reference_audio.wav> \
    -t "Text to synthesize" \
    -o output.wav

# v2.0 with emotion control
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r <reference_audio.wav> \
    -t "今天真是太开心了！" \
    -o output.wav \
    --emotion happy --emo-alpha 0.8
```

**Common Parameters:**
- `-m, --model`: MLX model directory path
- `-r, --ref-audio`: Reference audio file (.wav) or pre-computed speaker file (.npz)
- `-t, --text`: Text to synthesize
- `-o, --output`: Output audio file path
- `-q, --quantize`: Runtime quantization (4, 8, or fp32)
- `--seed`: Random seed for reproducible generation
- `-v, --verbose`: Show detailed timing info
- `--play`: Auto-play after generation
- `--memory-limit`: GPU memory limit in GB
- `--max-tokens`: Maximum mel tokens (default: 800 for v1.5, 1500 for v2.0)

**v2.0-only Parameters:**
- `--emotion`: Emotion control: happy/sad/angry/afraid/disgusted/melancholic/surprised/calm
- `--emo-alpha`: Emotion intensity 0.0-1.0 (default: 1.0, 0=reference audio emotion)
- `--steps`: Diffusion steps (default: 25)
- `--cfg`: CFG rate (default: 0.7)

### Pre-compute Speaker (v1.5 only, Faster Inference)

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

### Convert Model (Auto-detects version)

```bash
cd ~/Projects/mlx-indextts
uv sync --extra convert  # Install torch dependency

# v1.5
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    -o models/mlx-indexTTS-1.5

# v2.0
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-2 \
    -o models/mlx-indexTTS-2.0
```

## Available Models

| Model | Path | Status |
|-------|------|--------|
| IndexTTS 1.5 | `models/mlx-indexTTS-1.5` | ✅ |
| IndexTTS 2.0 | `models/mlx-indexTTS-2.0` | ✅ |

## Version Comparison

| Feature | v1.5 | v2.0 |
|---------|------|------|
| Sample rate | 24000 Hz | 22050 Hz |
| Max tokens | 800 | 1815 |
| Emotion control | ❌ | ✅ 8 emotions |
| Runtime quantization | ✅ | ✅ |
| RTF (M2 Max) | ~0.5 | ~1.3 |

## Supported Emotions (v2.0)

| English | Chinese |
|---------|---------|
| happy | 高兴 |
| angry | 愤怒 |
| sad | 悲伤 |
| afraid | 恐惧 |
| disgusted | 反感 |
| melancholic | 低落 |
| surprised | 惊讶 |
| calm | 自然 |

Mixed emotions: `--emotion "happy:0.6,sad:0.4"`

## Python API

```python
# v1.5
from mlx_indextts.generate import IndexTTS

tts = IndexTTS.load_model(
    "~/Projects/mlx-indextts/models/mlx-indexTTS-1.5",
    quantize_bits=8,
)
audio = tts.generate(text="Hello", ref_audio="./reference.wav")
tts.save_audio(audio, "./output.wav")

# v2.0
from mlx_indextts.generate_v2 import IndexTTSv2

tts = IndexTTSv2("~/Projects/mlx-indextts/models/mlx-indexTTS-2.0")
audio = tts.generate(
    text="Hello",
    reference_audio="./reference.wav",
    output_path="./output.wav",
    emotion="happy",
    emo_alpha=0.8,
)
```

## Common Tasks

### 1. Clone someone's voice (v1.5)
```bash
cd ~/Projects/mlx-indextts
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r person_recording.wav \
    -t "Content to speak" \
    -o output.wav \
    --quantize 8 --play
```

### 2. Generate with emotion (v2.0)
```bash
cd ~/Projects/mlx-indextts
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r person_recording.wav \
    -t "I'm so happy today!" \
    -o output.wav \
    --emotion happy --emo-alpha 0.8 --play
```

### 3. Ensure reproducible generation
```bash
cd ~/Projects/mlx-indextts
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r voice.wav \
    -t "Hello" \
    -o output.wav \
    --seed 42
```

## Performance

| Version | RTF (M2 Max) | Notes |
|---------|--------------|-------|
| v1.5 fp32 | ~0.5 | 2x faster than real-time |
| v1.5 8-bit | ~0.44 | Recommended |
| v2.0 fp32 | ~1.3 | With S2Mel CFM |
| v2.0 8-bit | ~1.1 | 1.2x speedup |

## Troubleshooting

### ModuleNotFoundError: No module named 'torch'
Model conversion requires torch:
```bash
cd ~/Projects/mlx-indextts
uv sync --extra convert
```

### v2.0-only parameters with v1.5 model
Error: "Parameters ['--emotion'] are only available for IndexTTS 2.0 models."
Solution: Use a v2.0 model for emotion control.

### Want faster inference?
1. Use `--quantize 8` for 8-bit quantization
2. For v1.5: Use pre-computed speaker (.npz) instead of raw audio (.wav)
