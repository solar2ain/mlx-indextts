# MLX-IndexTTS 项目

## 项目目标
实现 IndexTTS 的 MLX 版本，在 Apple Silicon 上通过 MLX 运行，无 PyTorch 运行时依赖。

## 环境要求
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 包管理器

## 版本支持
- Index-TTS 1.0: 原始版本 ✅
- IndexTTS 1.5: 改进版本 ✅
- IndexTTS 2.0: 最新版本（待实现）

## 当前状态
- **功能完整**: 所有核心组件已实现并通过测试
- **精度对齐**: 所有模块 MAE < 0.001，完美对齐 PyTorch
- **推理速度**: RTF ~0.5 (比实时快 2x，M3 Pro)
- **内存优化**: 默认 8GB 内存限制，支持运行时量化
- **CLI 可用**: `convert`, `generate`, `speaker` 命令

## 快速开始

```bash
# 安装 uv (如未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 (仅生成，不需要 torch)
uv sync

# 安装 (需要转换模型)
uv sync --extra convert

# 安装 (开发测试)
uv sync --extra dev

# 转换模型 (需要 --extra convert)
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    --output-dir models/mlx-indexTTS-1.5

# 生成语音
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -t "你好，世界！" \
    -o output.wav \
    --play  # 生成后直接播放

# 使用 8-bit 量化 (更快，更省内存)
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -t "你好，世界！" \
    -o output.wav \
    --quantize 8

# 使用随机种子确保可复现
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -t "你好，世界！" \
    -o output.wav \
    --seed 42

# 预计算 speaker 加速推理 (conditioning 计算加速 30x)
uv run mlx-indextts speaker \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -o ref_audios/voice_01.npz

# 使用预计算的 speaker 生成
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.npz \
    -t "你好，世界！" \
    -o output.wav

# 调整内存限制 (默认 8GB，0 表示不限制)
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -t "你好，世界！" \
    -o output.wav \
    --memory-limit 6
```

## 项目结构
```
mlx-indextts/
├── pyproject.toml
├── mlx_indextts/
│   ├── cli.py          # CLI (convert, generate, speaker)
│   ├── convert.py      # 模型转换
│   ├── generate.py     # 推理
│   ├── config.py       # 配置
│   ├── tokenizer.py    # 分词器
│   ├── mel.py          # Mel频谱
│   ├── normalize.py    # 文本正则化
│   └── models/
│       ├── gpt.py           # UnifiedVoice
│       ├── gpt2.py          # GPT-2 基础
│       ├── bigvgan.py       # BigVGAN
│       ├── conformer.py     # Conformer
│       ├── perceiver.py     # Perceiver
│       ├── ecapa_tdnn.py    # Speaker Encoder
│       ├── attention.py     # 注意力
│       └── activations.py   # 激活函数
├── tests/
├── ref_audios/             # 参考音频
├── scripts/                # 开发调试脚本
│   ├── dump_pytorch_outputs.py   # 导出 PyTorch 中间输出
│   ├── alignment_test.py         # PyTorch/MLX 逐层对齐测试
│   ├── compare_mlx_outputs.py    # 对比 MLX 输出与 PyTorch 中间结果
│   ├── memory_test.py            # PyTorch/MLX 内存使用对比
│   └── memory_test_mlx.py        # MLX Metal 内存详细测试
└── docs/
    └── indextts_1.5_alignment.md  # 1.5 版本对齐经验
```

## 模型架构

```
Audio Reference → [Conformer] → [Perceiver] → conditioning
                                                    ↓
Text → [Tokenizer] → text_tokens → [GPT] ← conditioning
                                      ↓
                                  mel_codes
                                      ↓
                    [BigVGAN + ECAPA-TDNN] → Audio Output
```

### 核心组件
| 组件 | 1.0 | 1.5 |
|------|-----|-----|
| GPT layers | 20 | 24 |
| model_dim | 1024 | 1280 |
| heads | 16 | 20 |
| max_mel_tokens | 605 | 800 |
| max_text_tokens | 402 | 600 |

## 参考资源
- 原始 index-tts 项目（PyTorch 实现）: `~/Projects/index-tts`，运行时注意使用 MPS。
- 原始 PyTorch 模型：
  - indexTTS-1.0：`~/Projects/index-tts/index-TTS`
  - indexTTS-1.5：`~/Projects/index-tts/indexTTS-1.5`
  - indexTTS-2.0：`~/Projects/index-tts/indexTTS-2`
- 测试参考音频: `ref_audios/`

## 开发说明

### 运行测试
```bash
uv run pytest -v
```

### MLX vs PyTorch 对齐
详细的对齐经验和踩坑记录见 [docs/indextts_1.5_alignment.md](docs/indextts_1.5_alignment.md)

### 关键 MLX 特性
- Conv1d 使用 NLC 格式 (batch, length, channels)
- Conv2d 使用 NHWC 格式
- 无 reflect padding，需用 numpy 替代
- BatchNorm 推理时必须调用 `.eval()`

## 模型文件

### 原始 PyTorch 模型
| 文件 | 推理需要 |
|------|---------|
| `gpt.pth` | ✅ |
| `bigvgan_generator.pth` | ✅ |
| `bpe.model` | ✅ |
| `config.yaml` | ✅ |
| `bigvgan_discriminator.pth` | ❌ |
| `dvae.pth` | ❌ |

### 转换后 MLX 模型
```
models/mlx-indexTTS-1.5/
├── gpt.safetensors        (~2.2GB)
├── bigvgan.safetensors    (511MB)
├── tokenizer.model        (465KB)
└── config.json            (1.9KB)
```

### 运行时量化
生成时使用 `--quantize` 参数启用量化（无需重新转换模型）：

| 量化 | 内存占用 | 速度 (RTF) | 质量 |
|------|----------|------------|------|
| fp32 (默认) | 基准 | ~0.52 | 最佳 |
| 8-bit | ~72% | ~0.44 | 良好 |
| 4-bit | ~60% | ~0.43 | 可接受 |

## 开发脚本

### 内存测试
```bash
# 对比 PyTorch 和 MLX 内存使用
uv run python scripts/memory_test.py

# 仅测试 MLX
uv run python scripts/memory_test.py --mlx

# MLX Metal 内存详细测试
uv run python scripts/memory_test_mlx.py
```

### 精度对齐测试
```bash
# 导出 PyTorch 中间输出 (需要在 index-tts 目录运行)
cd ~/Projects/index-tts
python ~/Projects/mlx-indextts/scripts/dump_pytorch_outputs.py

# PyTorch/MLX 逐层对齐测试
uv run python scripts/alignment_test.py

# 对比 MLX 输出与 PyTorch 中间结果
uv run python scripts/compare_mlx_outputs.py
```
