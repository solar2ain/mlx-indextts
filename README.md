# MLX-IndexTTS

IndexTTS for Apple Silicon using MLX. Zero-shot text-to-speech with voice cloning capabilities.

## Features

- Run IndexTTS 1.0/1.5 natively on Apple Silicon
- No PyTorch dependency at runtime
- RTF ~0.5 (2x faster than real-time on M3 Pro)
- Voice cloning from reference audio
- Pre-computed speaker conditioning for faster inference
- Reproducible generation with random seed

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX >= 0.18.0
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From source
git clone https://github.com/your-repo/mlx-indextts.git
cd mlx-indextts

# Basic install (generation only, no torch)
uv sync

# With model conversion support (requires torch)
uv sync --extra convert

# Development (includes torch + pytest)
uv sync --extra dev
```

## Quick Start

### 1. Convert Model

Convert PyTorch weights to MLX format:

```bash
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    --output ./models/mlx-indexTTS-1.5
```

### 2. Generate Speech

```bash
# Basic generation
uv run mlx-indextts generate \
    -m ./models/mlx-indexTTS-1.5 \
    -r ./reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o ./output.wav

# With 8-bit quantization (faster, lower memory)
uv run mlx-indextts generate \
    -m ./models/mlx-indexTTS-1.5 \
    -r ./reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o ./output.wav \
    --quantize 8
```

### 3. Pre-compute Speaker (Optional)

Pre-compute speaker conditioning for faster inference (30x speedup for conditioning):

```bash
# Save speaker conditioning
uv run mlx-indextts speaker \
    -m ./models/mlx-indexTTS-1.5 \
    -r ./reference.wav \
    -o ./speaker.npz

# Use pre-computed speaker
uv run mlx-indextts generate \
    -m ./models/mlx-indexTTS-1.5 \
    -r ./speaker.npz \
    -t "你好，世界！" \
    -o ./output.wav
```

### 4. Reproducible Generation

Use `--seed` for reproducible output:

```bash
uv run mlx-indextts generate \
    -m ./models/mlx-indexTTS-1.5 \
    -r ./reference.wav \
    -t "你好，世界！" \
    -o ./output.wav \
    --seed 42
```

## Python API

```python
from mlx_indextts import IndexTTS

# Load model
tts = IndexTTS.load_model("./models/mlx-indexTTS-1.5")

# Generate speech
audio = tts.generate(
    text="你好，这是一个语音合成测试。",
    ref_audio="./reference.wav",
    seed=42,  # Optional: for reproducible output
)

# Save audio
tts.save_audio(audio, "./output.wav")

# Pre-compute and save speaker for faster inference
tts.save_speaker("./reference.wav", "./speaker.npz")

# Use pre-computed speaker
audio = tts.generate(
    text="你好，世界！",
    ref_audio="./speaker.npz",  # Use .npz file
)
```

## CLI Options

```bash
uv run mlx-indextts generate --help

Options:
  -m, --model          Path to MLX model directory (required)
  -r, --ref-audio      Reference audio file or .npz speaker file (required)
  -t, --text           Text to synthesize (required)
  -o, --output         Output audio file path (required)
  --max-tokens         Maximum mel tokens to generate (default: 600)
  --temperature        Sampling temperature (default: 1.0)
  --top-k              Top-k sampling (default: 30)
  --top-p              Top-p sampling (default: 0.8)
  -s, --seed           Random seed for reproducible generation
  -q, --quantize       Runtime quantization: 4, 8, or fp32 (default: fp32)
  -v, --verbose        Print verbose output
  -p, --play           Play audio after generation
```

## Supported Models

| Model | Status | Notes |
|-------|--------|-------|
| IndexTTS 1.0 | ✅ | Original version |
| IndexTTS 1.5 | ✅ | Improved quality |
| IndexTTS 2.0 | 🚧 | Coming soon |

## Performance

| Metric | Value |
|--------|-------|
| RTF (M3 Pro) | ~0.5 |
| Conditioning from .wav | ~23ms |
| Conditioning from .npz | ~0.7ms |

### Quantization

Use `--quantize` for runtime quantization (faster inference, lower memory):

| Quantization | Memory | Speed (RTF) | Quality |
|--------------|--------|-------------|---------|
| fp32 (default) | Baseline | ~0.52 | Best |
| 8-bit | ~72% | ~0.44 | Good |
| 4-bit | ~60% | ~0.43 | Acceptable |

> **Note**: Quantization is applied at runtime, no need to re-convert models.

## Model Architecture

IndexTTS is a GPT-based zero-shot TTS model:

```
Reference Audio → [Conformer] → [Perceiver] → conditioning
                                                    ↓
Text → [Tokenizer] → [GPT Decoder] ← conditioning
                          ↓
                      mel_codes
                          ↓
              [BigVGAN + ECAPA-TDNN] → Audio
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- MLX >= 0.18.0
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## License

MIT License

## Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) - Original PyTorch implementation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
