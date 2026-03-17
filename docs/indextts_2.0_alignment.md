# MLX vs PyTorch 对齐经验 (IndexTTS 2.0)

本文档记录了将 IndexTTS 2.0 从 PyTorch 移植到 MLX 过程中遇到的所有对齐问题及解决方案。

## 精度对齐状态

| 模块 | MAE | 状态 | 备注 |
|------|-----|------|------|
| spk_cond_emb (W2V-BERT) | 0.0014 | ✅ | 完美对齐 |
| conditioning | 0.0011 | ✅ | 完美对齐 |
| forward_latent | 0.0014 | ✅ | 完美对齐 |
| gpt_layer | 0.00043 | ✅ | 完美对齐 |
| S_infer | 0.00043 | ✅ | 完美对齐 |
| Length Regulator | 0.000009 | ✅ | 完美对齐 |
| DiT | 0.025 | ✅ | 可接受 |
| BigVGAN v2 | 0.000001 | ✅ | 完美对齐 |

---

## 2.0 与 1.5 架构差异

| 特性 | 1.5 | 2.0 |
|------|-----|-----|
| max_mel_tokens | 800 | 1815 |
| 采样率 | 24000 Hz | 22050 Hz |
| BigVGAN | 自训练 (含 ECAPA-TDNN) | nvidia 预训练 (纯 vocoder) |
| 情感控制 | ❌ | ✅ 8类情感 |
| S2Mel (CFM) | ❌ | ✅ |
| Conformer input | mel (100) | W2V-BERT (1024) |

---

## 对齐关键点详解

### 1. Conformer input_size 差异

**问题**: 权重加载时形状不匹配
```
Expected: (512, 261632)
Got: (512, 25088)
```

**原因**:
- 1.5 的 Conformer `input_size=100` (mel 频带数)
- 2.0 的 Conformer `input_size=1024` (W2V-BERT 语义特征维度)
- PyTorch 2.0 模型硬编码了 `input_size=1024`

**解决**: 在 `gpt_v2.py` 中创建 ConformerConfig 时显式设置:
```python
ConformerConfig(input_size=1024, ...)  # 2.0 专用
```

### 2. S2Mel Dropout 推理时未禁用 ⚠️ 重复踩坑

**问题**: DiT 输出与 PyTorch 差异巨大 (MAE ~0.075)，多次运行结果不一致

**根因**: 加载 S2Mel 后没调用 `.eval()`，WaveNet 中的 Dropout 在推理时仍然激活

**现象**:
- `s2mel_mlx.training = True` (应该是 False)
- WaveNet 输出有 ~20% 的零值 (Dropout 在工作)
- 每次推理结果不同 (非确定性)

**解决**: 所有模型加载后必须调用 eval():
```python
self.s2mel_mlx = S2Mel()
self.s2mel_mlx.load_weights(path)
self.s2mel_mlx.eval()  # 关键！
```

**教训**: 这是 1.5 版本 BatchNorm 问题的重复。**所有模型加载后必须调用 `.eval()`**

### 3. ref_mel 使用 librosa 导致幅度异常

**问题**: 生成音频幅度是 PyTorch 的 ~3 倍

| 指标 | MLX (修复前) | PyTorch |
|------|-------------|---------|
| wav std | 0.39 | 0.11 |
| 幅度比 | **3.5x** | 1.0x |

**根因**: `generate_v2.py` 中使用 librosa 计算 `ref_mel`（送给 CFM 作为 prompt），但 librosa 和 PyTorch 的 STFT 实现有细微差异:

| 实现 | mean | std | max |
|------|------|-----|-----|
| **PyTorch (正确)** | -4.32 | 1.74 | **1.40** |
| **librosa (错误)** | -5.23 | 3.37 | **5.97** |

主要差异:
1. PyTorch 使用 `reflect` padding 后再 STFT
2. librosa 的 `melspectrogram` 内部处理方式不同
3. 导致 mel 谱的动态范围差异巨大

**解决**: 将 librosa 实现替换为 PyTorch 实现（与 index-tts 完全一致）:
```python
# 修复前 (librosa)
mel = librosa.feature.melspectrogram(y=audio_np, ...)
mel = torch.log(torch.clamp(mel, min=1e-5))

# 修复后 (PyTorch, 与 index-tts 一致)
audio = torch.nn.functional.pad(audio.unsqueeze(1), (pad_size, pad_size), mode="reflect")
spec = torch.stft(audio, n_fft, hop_length, win_length, window=hann_window, ...)
spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
spec = torch.matmul(mel_basis, spec)
spec = torch.log(torch.clamp(spec, min=1e-5))
```

**注意**: 1.5 版本不受影响，因为 1.5 使用自实现的 `MelSpectrogramExtractor`（纯 MLX）

### 4. GPT mel position embedding 索引错误

**问题**: 生成音频幅度异常（约 3 倍）

**根因**: GPT 自回归生成时，mel position embedding 的索引计算错误

PyTorch 使用 `attention_mask.shape[1] - mel_len` 计算位置:
- start_mel_token: position 0
- 第一个生成的 token: position 2 (因为 attention_mask 从 mel_len+1 开始)
- 第二个生成的 token: position 3
- ...

MLX 错误地使用 `len(mel_codes)` 计算位置:
- start_mel_token: position 0
- 第一个生成的 token: position 1 ❌ (应该是 2)
- 第二个生成的 token: position 2 ❌ (应该是 3)

**解决**: 在 `generate.py` 和 `generate_v2.py` 中:
```python
# 修复前
mel_pos = len(mel_codes)

# 修复后
mel_pos = len(mel_codes) + 1
```

**验证** (多步 GPT logits 对比):
- 修复前: Step 1-9 MAE > 0.7，top-5 overlap 0-4/5
- 修复后: Step 0-9 MAE < 0.005，top-5 overlap **5/5** 全部完美匹配

### 5. CLI max_tokens 默认值

**问题**: 2.0 长文本被截断，但 1.5 正常

**原因**: CLI 统一使用 `max_tokens=800` 作为默认值

| 版本 | 配置 max_mel_tokens | CLI 默认 | 结果 |
|------|---------------------|----------|------|
| 1.5 | 800 | 800 | ✅ 刚好 |
| 2.0 | 1815 | 800 | ❌ 只有 44% |

**解决**: CLI 根据版本设置不同默认值:
```python
max_tokens = 1500 if version == "2.0" else 800
```

### 6. 输入格式约定

**2.0 与 1.5 的差异**:
- 1.5: `get_conditioning` 接收 mel spectrogram (batch, time, n_mels)
- 2.0: `get_conditioning` 接收 W2V-BERT 特征 (batch, 1024, time) NCL 格式

**关键**: PyTorch `get_conditioning` 内部会做 transpose，所以 MLX 也要在内部做 transpose:
```python
# PyTorch v2 行为:
speech_conditioning_input, mask = self.conditioning_encoder(
    speech_conditioning_input.transpose(1, 2),  # NCL -> NLC
    cond_mel_lengths
)
```

### 7. LengthRegulator 权重前缀问题

**问题**: 使用 `load_weights(path)` 加载 safetensors 后，权重实际没有被赋值

**原因**: 权重文件的 key 带有 `length_regulator.xxx` 前缀，但单独创建的 `InterpolateRegulator` 实例寻找的是 `xxx` (无前缀)

**解决**: 加载权重时需要 strip 前缀:
```python
weights = mx.load(weights_path)
lr_weights = {k.replace('length_regulator.', ''): v
              for k, v in weights.items()
              if k.startswith('length_regulator.')}
lr.load_weights(lr_weights, strict=True)
```

**验证**: 用 `strict=True` 确保所有权重都被正确加载

### 8. MLX Conv1d 格式处理

**关键**: MLX Conv1d 期望 NLC 格式，而 PyTorch 使用 NCL 格式

**正确做法** (参考 BigVGAN v2):
```python
# NCL 输入 -> 转换为 NLC -> Conv1d -> 转换回 NCL
x = x.transpose(0, 2, 1)  # NCL -> NLC
x = conv(x)
x = x.transpose(0, 2, 1)  # NLC -> NCL
```

**GroupNorm**: 仍然使用 NCL 格式处理

### 9. FinalLayer 的 LayerNorm

**问题**: PyTorch 使用 `nn.LayerNorm(hidden_size, elementwise_affine=False)`

**关键**: `elementwise_affine=False` 意味着没有可学习的 weight/bias 参数

**解决**: MLX 实现中不要创建 weight/bias，只做标准化:
```python
def _layer_norm(self, x):
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    return (x - mean) / mx.sqrt(var + 1e-6)
```

---

## 权重转换注意事项

### 复用 1.5 转换逻辑

**教训**: GPT v2 转换时从头写了新的转换脚本，导致大量重复踩坑（key 映射、权重转置等）

**正确做法**:
- 1.5 的 `mlx_indextts/convert.py` 已经处理好了所有 backbone 的权重转换
- 应该直接复用 `convert_gpt_weights()` 函数，只需为 2.0 新增模块添加处理逻辑
- **先读 1.5 对齐文档中的踩坑记录**

### 扩展现有转换逻辑

修改 `convert.py` 时只添加对新模块的支持，不改变现有逻辑:
```python
# 原来只处理 perceiver_encoder
if "perceiver_encoder" in key:

# 扩展为同时处理 emo_perceiver_encoder (逻辑相同)
if "perceiver_encoder" in key or "emo_perceiver_encoder" in key:
```

**验证**: 修改后运行 `pytest tests/` 确保 1.5 测试仍然通过

---

## 调试技巧

### 逐层对比
1. 保存 PyTorch 中间输出到 numpy (`scripts/dump_pytorch_outputs_v2.py`)
2. 在 MLX 中加载相同输入
3. 计算 MAE 找到差异点

### 常见问题排查

| 现象 | 可能原因 |
|------|---------|
| 输出全为 0 或不一致 | Dropout/BatchNorm 未切换到 eval 模式 |
| 音频幅度异常 (3x) | mel spectrogram 实现不一致 (librosa vs PyTorch) |
| 音频幅度异常 (3x) | GPT position embedding 索引错误 |
| 权重加载后值不变 | 权重 key 前缀不匹配 |
| 形状不匹配 | Conformer input_size 配置错误 |
| 长文本截断 | max_tokens 默认值太小 |

---

## 开发检查清单

实现新模块前:

- [ ] 读取 `docs/indextts_1.5_alignment.md` (1.5 踩坑经验)
- [ ] 读取本文档 (2.0 踩坑经验)
- [ ] 检查现有代码是否可复用 (`convert.py`, `models/`)
- [ ] 对比 PyTorch 源码的初始化参数
- [ ] 导出 PyTorch 中间结果用于对齐测试
- [ ] **所有模型加载后调用 `.eval()`**
- [ ] 修改公共代码后运行 `pytest tests/`
