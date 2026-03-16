# MLX-IndexTTS 项目

## 项目目标
实现 IndexTTS 的 MLX 版本，在 Apple Silicon 上通过 MLX 运行，无 PyTorch 运行时依赖。

## 版本支持
- Index-TTS 1.0: 原始版本 ✅
- IndexTTS 1.5: 改进版本 ✅
- IndexTTS 2.0: 最新版本（待实现）

## 当前状态
- **功能完整**: 所有核心组件已实现并通过测试
- **精度对齐**: 所有模块 MAE < 0.001，完美对齐 PyTorch
- **推理速度**: RTF ~0.5 (比实时快 2x，M3 Pro)
- **CLI 可用**: `convert`, `generate`, `speaker` 命令

## 快速开始

```bash
# 安装 (仅生成，不需要 torch)
uv sync

# 安装 (需要转换模型)
uv sync --extra convert

# 安装 (开发测试)
uv sync --extra dev

# 转换模型 (需要 --extra convert)
mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    --output-dir models/mlx-indexTTS-1.5

# 生成语音
mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -t "你好，世界！" \
    -o output.wav \
    --play  # 生成后直接播放

# 使用随机种子确保可复现
mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -t "你好，世界！" \
    -o output.wav \
    --seed 42

# 预计算 speaker 加速推理 (conditioning 计算加速 30x)
mlx-indextts speaker \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -o ref_audios/voice_01.npz

# 使用预计算的 speaker 生成
mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.npz \
    -t "你好，世界！" \
    -o output.wav
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
│   └── compare_mlx_outputs.py    # 对比 MLX 输出与 PyTorch 中间结果
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
- 原始 index-tts 项目: `/Users/didi/Projects/index-tts`
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
├── gpt.safetensors        (2.2GB)
├── bigvgan.safetensors    (511MB)
├── tokenizer.model        (465KB)
└── config.json            (1.9KB)
```
