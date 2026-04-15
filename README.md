# MLX-IndexTTS

IndexTTS for Apple Silicon using MLX. Zero-shot text-to-speech with voice cloning capabilities.

## Features

- Run IndexTTS 1.5/2.0 natively on Apple Silicon
- RTF ~0.5 (2x faster than real-time on M2 Max)
- Voice cloning from reference audio
- **v2.0**: Emotion control (8 emotions)
- Auto-detect model version (1.5/2.0)

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/user/mlx-indextts.git
cd mlx-indextts

# Basic install (generation only)
uv sync

# With model conversion support (requires torch)
uv sync --extra convert
```

## Quick Start

### 1. Convert Model (auto-detects version)

```bash
# Convert IndexTTS 1.5
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    -o models/mlx-indexTTS-1.5

# Convert IndexTTS 2.0
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-2 \
    -o models/mlx-indexTTS-2.0
```

### 2. Generate Speech (auto-detects version)

```bash
# v1.5
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o output.wav

# v2.0
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -t "你好，这是一个语音合成测试。" \
    -o output.wav

# v2.0 with emotion control
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -t "今天真是太开心了！" \
    -o output.wav \
    --emotion happy --emo-alpha 0.6
```

### 3. Pre-compute Speaker (Faster Inference)

Pre-compute speaker conditioning to skip audio preprocessing on subsequent generations.

```bash
# v1.5
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-1.5 \
    -r reference.wav \
    -o speaker_v15.npz

# v2.0
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-2.0 \
    -r reference.wav \
    -o speaker_v20.npz

# Use pre-computed speaker (much faster loading)
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r speaker_v20.npz \
    -t "你好，世界！" \
    -o output.wav
```

**Note**: v1.5 and v2.0 speaker files are incompatible - each version requires its own .npz file.

## Python API

```python
# v1.5
from mlx_indextts.generate import IndexTTS

tts = IndexTTS.load_model("models/mlx-indexTTS-1.5")
audio = tts.generate(text="你好", ref_audio="reference.wav")
tts.save_audio(audio, "output.wav")

# v2.0
from mlx_indextts.generate_v2 import IndexTTSv2

tts = IndexTTSv2("models/mlx-indexTTS-2.0")
audio = tts.generate(
    text="你好",
    reference_audio="reference.wav",
    output_path="output.wav",
    emotion="happy",
    emo_alpha=0.6,
)
```

## CLI Options

```
mlx-indextts generate [OPTIONS]

Required:
  -m, --model        Model directory
  -r, --ref-audio    Reference audio (.wav or .npz)
  -t, --text         Text to synthesize
  -o, --output       Output file

Common options:
  --max-tokens       Max mel tokens (default: 800 for v1.5, 1500 for v2.0)
  --temperature      Sampling temperature (default: 1.0 for v1.5, 0.8 for v2.0)
  --seed, -s         Random seed for reproducibility
  -v, --verbose      Verbose output
  -p, --play         Play audio after generation
  --quantize, -q     Runtime quantization: 4, 8, or fp32

v2.0 only:
  --emotion          Emotion: happy/sad/angry/afraid/disgusted/melancholic/surprised/calm
  --emo-alpha        Emotion intensity 0.0-1.0 (default: 0.6, recommend ≤ 0.8)
  --diffusion-steps  Diffusion steps (default: 25)
  --cfg-rate         CFG rate (default: 0.7)
```

## Version Comparison

| Feature | v1.5 | v2.0 |
|---------|------|------|
| Sample rate | 24000 Hz | 22050 Hz |
| Max tokens | 800 | 1815 |
| Default temperature | 1.0 | 0.8 |
| Emotion control | ❌ | ✅ 8 emotions |
| S2Mel (CFM) | ❌ | ✅ |
| BigVGAN | Custom | nvidia pretrained |
| Runtime quantization | ✅ | ✅ |
| Speaker pre-compute | ✅ | ✅ |

## Supported Emotions (v2.0)

| English | 中文 |
|---------|------|
| happy | 高兴 |
| angry | 愤怒 |
| sad | 悲伤 |
| afraid | 恐惧 |
| disgusted | 反感 |
| melancholic | 低落 |
| surprised | 惊讶 |
| calm | 自然 |

Mixed emotions: `--emotion "happy:0.6,sad:0.4"`

## Performance

| Metric | v1.5 | v2.0 |
|--------|------|------|
| RTF (M2 Max) | ~0.5 | ~1.3 |
| Load time (.wav) | ~0.3s | ~9s |
| Load time (.npz) | ~0.3s | ~1.5s |

## License

MIT License

## Acknowledgments

- [IndexTTS](https://github.com/index-tts/index-tts) - Original PyTorch implementation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
