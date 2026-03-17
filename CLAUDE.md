# MLX-IndexTTS 项目

## 项目目标
实现 IndexTTS 的 MLX 版本，在 Apple Silicon 上高效运行。

## 环境要求
- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) 包管理器

## 版本支持
- IndexTTS 1.5: ✅ 完成
- IndexTTS 2.0: ✅ 完成 (含情感控制)

## 快速开始

```bash
# 安装
uv sync                    # 仅生成
uv sync --extra convert    # 需要转换模型

# 转换模型 (自动识别版本)
uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-1.5 \
    -o models/mlx-indexTTS-1.5

uv run mlx-indextts convert \
    --model-dir /path/to/indexTTS-2 \
    -o models/mlx-indexTTS-2.0

# 生成语音 (自动识别版本)
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-1.5 \
    -r ref_audios/voice_01.wav \
    -t "你好，世界！" \
    -o output.wav

# 2.0 情感控制
uv run mlx-indextts generate \
    -m models/mlx-indexTTS-2.0 \
    -r ref_audios/voice_01.wav \
    -t "今天真开心！" \
    -o output.wav \
    --emotion happy --emo-alpha 0.8
```

## 项目结构
```
mlx-indextts/
├── mlx_indextts/
│   ├── cli.py              # CLI 入口 (统一 1.5/2.0)
│   ├── convert.py          # 1.5 模型转换
│   ├── convert_v2.py       # 2.0 模型转换
│   ├── generate.py         # 1.5 推理
│   ├── generate_v2.py      # 2.0 推理
│   ├── config.py           # 配置
│   ├── tokenizer.py        # 分词器
│   ├── mel.py              # Mel频谱
│   ├── normalize.py        # 文本正则化
│   ├── indextts/           # PyTorch 依赖模块 (2.0 运行时)
│   └── models/
│       ├── # === 共享 (1.5 & 2.0) ===
│       ├── gpt2.py, conformer.py, perceiver.py, attention.py, activations.py
│       ├── # === 1.5 专属 ===
│       ├── gpt.py, bigvgan.py, ecapa_tdnn.py
│       ├── # === 2.0 专属 ===
│       ├── gpt_v2.py, bigvgan_v2.py
│       └── s2mel/           # S2Mel (DiT + CFM)
├── tests/                   # pytest 测试
├── scripts/                 # 开发调试脚本
├── ref_audios/              # 参考音频
└── docs/
    ├── indextts_1.5_alignment.md  # 1.5 踩坑经验
    └── indextts_2.0_progress.md   # 2.0 总结
```

## 版本差异

| 特性 | 1.5 | 2.0 |
|------|-----|-----|
| max_mel_tokens | 800 | 1815 |
| 采样率 | 24000 Hz | 22050 Hz |
| 情感控制 | ❌ | ✅ 8类 |
| S2Mel (CFM) | ❌ | ✅ |
| 运行时量化 | ✅ | ❌ |

## 参考资源
- PyTorch 原始项目: `~/Projects/index-tts`
- PyTorch 模型:
  - 1.5: `~/Projects/index-tts/indexTTS-1.5`
  - 2.0: `~/Projects/index-tts/indexTTS-2`

## 开发说明

### ⚠️ 重要：开始前必读

1. **[docs/indextts_1.5_alignment.md](docs/indextts_1.5_alignment.md)** - MLX 框架特性、权重转换
2. **[docs/indextts_2.0_alignment.md](docs/indextts_2.0_alignment.md)** - 2.0 踩坑经验

### 核心原则
- 优先复用现有代码，不要从头重写
- 所有模型加载后必须调用 `.eval()`
- 修改公共代码后运行 `uv run pytest tests/`

### MLX 关键特性
- Conv1d: NLC 格式 (batch, length, channels)
- Conv2d: NHWC 格式
- 无 reflect padding，需用 numpy
- BatchNorm/Dropout 推理时必须 `.eval()`

### 运行测试
```bash
uv run pytest -v
```

## 模型文件

### 1.5 转换后
```
models/mlx-indexTTS-1.5/
├── gpt.safetensors        (~2.2GB)
├── bigvgan.safetensors    (511MB)
├── tokenizer.model
└── config.json
```

### 2.0 转换后
```
models/mlx-indexTTS-2.0/
├── gpt.safetensors        (3.3GB)
├── s2mel.safetensors      (395MB)
├── bigvgan.safetensors    (428MB)
├── tokenizer.model
├── config.json
├── feat1.pt               (spk_matrix)
└── feat2.pt               (emo_matrix)
```

## 开发脚本

```bash
# 精度对齐测试
uv run python scripts/alignment_test.py      # 1.5
uv run python scripts/alignment_test_v2.py   # 2.0

# 生成测试音频
uv run python scripts/generate_test_audio.py --mlx-only
```
