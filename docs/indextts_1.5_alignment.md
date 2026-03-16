# MLX vs PyTorch 对齐经验 (IndexTTS 1.0/1.5)

本文档记录了将 IndexTTS 从 PyTorch 移植到 MLX 过程中遇到的所有对齐问题及解决方案。

## 精度对齐状态

| 模块 | MAE | 状态 | 备注 |
|------|-----|------|------|
| Mel Spectrogram | 0.000002 | ✅ | 完美对齐 |
| Text Tokenization | 完全匹配 | ✅ | - |
| ECAPA-TDNN | 0.000000 | ✅ | 修复 reflect padding |
| Speaker Embedding | 0.000000 | ✅ | 完美对齐 |
| BigVGAN Output | 0.000000 | ✅ | 完美对齐 |
| Conformer Encoder | 0.000007 | ✅ | 修复 rel_shift |
| Perceiver Encoder | 0.000027 | ✅ | 完美对齐 |
| GPT-2 hidden state | 0.000000 | ✅ | 完美对齐 |

## 性能优化历史

| 优化项 | 优化前 | 优化后 | 加速比 |
|--------|--------|--------|--------|
| Activation1d (numpy→MLX groups conv) | RTF 5.8 | RTF 0.5 | ~11x |
| ECAPA-TDNN reflect padding | 边界 MAE 0.65 | MAE 0.000000 | - |
| Conformer RelPosAttn (移除 rel_shift) | MAE 18 | MAE 0.001564 | - |

---

## MLX 框架特性

### 数据格式约定
- **Conv1d**: NLC 格式 `(batch, length, channels)` - 与 PyTorch NCL 不同
- **Conv2d**: NHWC 格式 `(batch, height, width, channels)` - 与 PyTorch NCHW 不同
- 权重转换时需要 transpose

### 不支持的特性
- 无 reflect padding，需用 numpy 替代
- 无 `mx.zeros_like(dtype=...)` 语法
- 无 JAX 风格的 `.at[].set()` 语法

---

## 对齐关键点详解

### 1. BatchNorm 训练/推理模式
- **问题**: MLX BatchNorm 默认 `training=True`，会计算 batch 统计量
- **解决**: 推理时必须调用 `model.eval()` 或 `module.eval()` 切换到推理模式
- **影响**: 不调用 eval() 会导致输出值完全错误

### 2. ECAPA-TDNN 数据格式
- **PyTorch ECAPA**: 外部接口 NLC `(batch, time, n_mels)`，内部 forward 做 transpose 转为 NCL
- **BigVGAN 调用**: `bigvgan(latent, mel_ref.transpose(1, 2))` - mel_ref 从 NCL 转为 NLC 传入
- **MLX ECAPA**: 自动检测格式 - 如果 `shape[1] == input_size` 则为 NCL，否则 NLC

### 3. TDNNBlock 操作顺序
- **正确顺序**: conv → activation (ReLU) → norm
- **PyTorch**: `self.norm(self.activation(self.conv(x)))`
- **错误实现**: conv → norm → relu (会导致输出差异)

### 4. TDNNBlock padding (ECAPA-TDNN)
- **PyTorch**: 使用 `padding_mode="reflect"` (SpeechBrain Conv1d)
- **MLX**: 必须手动实现 reflect padding (MLX Conv1d 只支持 zero padding)
- **实现**: 用 numpy `np.pad(..., mode='reflect')` 后转回 mx.array
- **影响**: 修复后 ECAPA-TDNN 所有层 MAE 降至 0.000000

### 5. AttentiveStatisticsPooling
- **关键**: 在 TDNN 和 conv 之间有 tanh 激活
- **PyTorch**: `attn = self.conv(self.tanh(self.tdnn(attn)))`
- **遗漏 tanh 会导致输出差异**

### 6. mx.topk 返回顺序
- **MLX**: `mx.topk(x, k)` 返回**升序**排列的 top-k 值（第一个是最小的）
- **PyTorch**: `torch.topk(x, k)` 返回**降序**排列
- **采样时**: 阈值应取 `top_k_values[:, :1]`（最小的 top-k 值）

### 7. Snake/SnakeBeta 激活函数
- alpha/beta 是 1D tensor `(channels,)`
- 在 forward 中需要 unsqueeze 到 `(1, channels, 1)` 用于 NCL 格式广播
- 期望输入格式: NCL `(batch, channels, length)`

### 8. AttentiveStatisticsPooling 数值稳定性
- **问题**: `variance = E[x²] - E[x]²` 可能因浮点精度产生微小负值
- **解决**: 在 sqrt 前使用 `mx.maximum(variance, 0.0)` clip 负值
- **不处理会导致 NaN**

### 9. GPT-2 Causal Mask
- **问题**: GPT-2 是 autoregressive 模型，必须使用 causal mask
- **PyTorch**: HuggingFace GPT2Model 内部自动应用 causal mask
- **MLX**: 需要在 GPT2Model.__call__ 中显式创建和应用 causal mask
- **Mask 生成**: 使用 `mx.where(mask > 0, float("-inf"), 0.0)` 而非 `mask * float("-inf")`（后者产生 NaN）

### 10. GPT-2 Attention Bias
- **问题**: HuggingFace GPT2 的 c_attn 和 c_proj 都有 bias
- **转换注意**: 不要跳过 `c_attn.bias`，只跳过 causal mask buffer（`attn.bias`）
- **错误**: `if "attn.bias" in key: continue` 会误跳过 `c_attn.bias`
- **正确**: `if key.endswith("attn.bias") and "c_attn" not in key: continue`

### 11. GPT forward_latent Token 处理
- **PyTorch**: text 和 mel 都是 `[start, ..., stop]` 格式
- **关键**: mel_codes 末尾必须加 stop token
- **返回**: 从 hidden 中提取 mel_len 个 latent（跳过 start，不含 stop）

### 12. BigVGAN Activation1d 抗锯齿
- **PyTorch**: `Activation1d` 包含 upsample → SnakeBeta → downsample 的完整抗锯齿处理
- **MLX 必须复刻**: 不是可选的，否则输出范围会错误
- **实现**: 使用 `mx.conv1d(..., groups=C)` 和 `mx.conv_transpose1d(..., groups=C)` 做 depthwise 卷积
- **Padding**: 用 numpy `np.pad(..., mode='edge')` 做 replicate padding
- **验证**: 抗锯齿后 BigVGAN 输出 MAE 降至 0.000000

### 13. BigVGAN AMPBlock 数据格式
- **输入/输出**: NCL 格式 `(batch, channels, length)`
- **内部 Conv1d**: 需要转换到 NLC 再转回 NCL
- **SnakeBeta**: 期望 NCL 格式

### 14. Activation1d 性能优化
- **问题**: 最初用 numpy for 循环实现，RTF ~5.8，GPU 空闲
- **原因**: MLX 不支持 `.at[].add()` 语法
- **解决**: 使用 `mx.conv_transpose1d(..., groups=C)` 替代手动循环
- **结果**: RTF 降至 ~0.5，GPU 正常使用

### 15. Conformer RelPositionMultiHeadAttention
- **问题**: PyTorch 原版注释掉了 rel_shift，MLX 实现误用
- **PyTorch**: `# matrix_bd = self.rel_shift(matrix_bd)` (注释掉，不使用)
- **MLX 错误**: 调用了 `_rel_shift(matrix_bd)`，导致 MAE=18
- **Position bias 顺序**: PyTorch 在 transpose **前** 加 bias
  ```python
  # 正确: bias 加到 q 后再 transpose
  q_with_bias_u = (q + self.pos_bias_u).transpose(0, 2, 1, 3)
  # 错误: transpose 后再加 bias
  q = q.transpose(0, 2, 1, 3)
  q_with_bias_u = q + self.pos_bias_u  # 维度不匹配或语义错误
  ```
- **修复后**: Conformer MAE 从 18 降至 0.000007

### 16. Mel Position Encoding in Autoregressive Loop (重要!)
- **问题**: 生成的语音中间漏字
- **原因**: 使用了全局序列位置而非 mel 相对位置
- **错误代码**:
  ```python
  pos = input_emb.shape[1] + len(mel_codes) - 1  # 可能是 65+，超出 mel_pos_embedding 范围
  last_emb = last_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(pos)
  ```
- **PyTorch 行为**: `attention_mask.shape[1] - mel_len`，即相对于 mel 序列起点的位置
- **正确代码**:
  ```python
  mel_pos = len(mel_codes)  # start_mel_token 位置是 0，第一个生成的 token 位置是 1
  last_emb = last_emb + self.gpt.mel_pos_embedding.get_fixed_embedding(mel_pos)
  ```
- **修复后**: 生成质量完全正常，中英文语音完整无漏字

---

## 权重转换注意事项

### Conv1d 权重
```python
# PyTorch: (out_channels, in_channels, kernel_size)
# MLX:     (out_channels, kernel_size, in_channels)
mlx_weight = pt_weight.transpose(0, 2, 1)
```

### ConvTranspose1d 权重
```python
# PyTorch: (in_channels, out_channels, kernel_size)
# MLX:     (out_channels, kernel_size, in_channels)
mlx_weight = pt_weight.transpose(1, 2, 0)
```

### GPT-2 权重
```python
# Attention/MLP weights need transpose
mlx_weight = pt_weight.T
```

### 需要跳过的权重
- `num_batches_tracked` (BatchNorm 统计)
- `attn.bias` (causal mask buffer, 但不要跳过 `c_attn.bias`)

---

## 调试技巧

### 逐层对比
1. 保存 PyTorch 中间输出到 numpy
2. 在 MLX 中加载相同输入
3. 计算 MAE 找到差异点

### 常见问题排查
| 现象 | 可能原因 |
|------|---------|
| 输出全为 0 | BatchNorm 未切换到 eval 模式 |
| 边界 MAE 高 | padding 方式不一致 |
| NaN 输出 | 负数开方、-inf 乘 0 |
| 数值范围错误 | 激活函数实现不正确 |
| 生成漏字 | 位置编码计算错误 |
