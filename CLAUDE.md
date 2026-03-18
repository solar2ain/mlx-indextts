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
    └── indextts_2.0_alignment.md  # 2.0 踩坑经验
```

## 版本差异

| 特性 | 1.5 | 2.0 |
|------|-----|-----|
| max_mel_tokens | 800 | 1815 |
| 采样率 | 24000 Hz | 22050 Hz |
| 默认 temperature | 1.0 | 0.8 |
| 情感控制 | ❌ | ✅ 8类 |
| S2Mel (CFM) | ❌ | ✅ |
| 运行时量化 | ✅ | ✅ |
| Speaker 预计算 | ✅ | ✅ |
| RTF (M2 Max) | ~0.5 | ~1.3 |
| 加载时间 (.npz) | ~0.3s | ~1.5s |

## 参考资源
- PyTorch 原始项目: `~/Projects/index-tts`
- PyTorch 模型:
  - 1.5: `~/Projects/index-tts/indexTTS-1.5`
  - 2.0: `~/Projects/index-tts/indexTTS-2`

## 采样策略分析 (PyTorch vs MLX)

### PyTorch 原始实现
PyTorch IndexTTS 使用 HuggingFace `generate()`:
```python
gen = self.gpt.gpt.generate(
    inputs_embeds=emb,
    do_sample=True,        # 启用采样
    max_new_tokens=max_mel_tokens,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    repetition_penalty=repetition_penalty,  # 默认 10.0
    num_beams=num_beams,   # 默认 3
    length_penalty=length_penalty,
)
```
- `do_sample=True` + `num_beams>1` = **beam-sample** (并行采样 + beam 选择)
- `repetition_penalty=10.0` 强力惩罚重复 token

### MLX 当前实现
MLX 使用 **纯采样** (top-k + top-p + repetition_penalty):
```python
# 不使用 beam search，但实现了 repetition_penalty
next_token = self._sample(logits, temperature, top_k, top_p,
                          repetition_penalty, generated_tokens)
```

### 差异影响
| 特性 | PyTorch | MLX | 影响 |
|------|---------|-----|------|
| 采样方式 | beam-sample | 纯采样 | MLX 输出多样性更高 |
| repetition_penalty | ✅ 10.0 | ✅ 10.0 | 已对齐 |
| num_beams | 3 | 1 | MLX 无 beam 选择 |
| length_penalty | 1.0 | ❌ | MLX 无长度偏好 |

### Beam Search 复杂度分析
实现 beam search 需要:
1. **多 beam KV cache 管理** - 每个 beam 独立 cache
2. **分数累积与排序** - log_prob 累积 + length_penalty
3. **beam 剪枝与扩展** - top-k beams 选择
4. **early stopping** - 所有 beam 到达 EOS

**结论**: 当前 `repetition_penalty=10.0` 已能有效避免重复，beam search 收益有限，暂不实现。

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
├── vq2emb.safetensors     (0.3MB)  # MLX vq2emb
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
