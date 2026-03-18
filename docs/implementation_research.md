# MLX 实现研究笔记

本文档记录 MLX-IndexTTS 实现过程中的技术调研和分析结果。

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

---

## Batch 推理分析 (infer_fast)

PyTorch IndexTTS 1.5 提供 `infer_fast` 方法，通过 batch 处理多个 segments 加速长文本生成。

### 核心机制
1. **Bucket 分组**: 将相似长度的 segments 分组，减少 padding 浪费
2. **Batch GPT generate**: 利用 HuggingFace `generate()` 的 batch 能力
3. **Batch BigVGAN**: 合并多个 latents 一起解码

### MLX 移植难度
- **高**: GPT generate 需重写支持 batch + 动态长度 + early stopping
- **中**: KV cache 需支持 batch 维度
- **低**: BigVGAN batch 化相对简单

### 性能与显存预估
| 指标 | 当前 | batch=4 |
|------|------|---------|
| v1.5 RTF | ~0.5 | ~0.3-0.35 (1.4-1.7x) |
| v2.0 RTF | ~1.3 | ~0.9-1.0 (1.3-1.4x) |
| 显存增加 | - | +100% |

**结论**: 短文本无收益，长文本加速 1.3-1.7x，但显存翻倍。暂不移植。

---

## MLX-LM Batch 能力调研

mlx-lm 已有完整的 batch generation 支持 (`BatchGenerator`, `BatchKVCache`)。

### 可直接复用的组件
来自 `mlx_lm.sample_utils`:
- `make_sampler(temp, top_k, top_p, min_p)` - 采样器工厂
- `make_logits_processors(repetition_penalty)` - logits 处理器
- `apply_top_k`, `apply_top_p`, `categorical_sampling` - 采样函数

### 不能直接复用的组件
| 组件 | 原因 |
|------|------|
| `BatchGenerator` | 假设 model 接受 token ids，IndexTTS 需要自定义 embedding 流程 |
| `BatchKVCache` | cache 结构不同 (head 维度位置) |
| `generate_step` | 耦合了 model 调用方式 |

### 根本原因
IndexTTS GPT 有特殊流程，与标准 LLM 的 token-in/logits-out 模式不兼容:
```
conditioning → text_emb → mel_emb + pos_emb → forward → mel_head
```

### 建议
先复用 sampling 工具函数简化代码，batch 推理需自行实现。

---

---

## mlx-audio 采样实现调研

mlx-audio 项目中有 IndexTTS 1.5 和 2.0 的实现，调研其采样策略实现。

### 依赖关系
mlx-audio 直接使用 mlx-lm 的采样工具:
```python
from mlx_lm.sample_utils import make_sampler
```

### 各版本实现对比

| 特性 | mlx-audio indextts | mlx-audio indextts2 | 本项目实现 |
|------|-------------------|--------------------| ---------|
| temperature | ✅ 0.8 | ✅ configurable | ✅ configurable |
| top_k | ✅ 30 | ✅ 30 | ✅ configurable |
| top_p | ❌ | ❌ | ✅ configurable |
| repetition_penalty | ❌ | ✅ custom | ✅ 10.0 |
| @mx.compile | ✅ (via make_sampler) | ✅ (via make_sampler) | ❌ |

### mlx-audio indextts v1.5
```python
# 简单实现，缺少 top_p 和 repetition_penalty
sampler = make_sampler(temp=0.8, top_k=30)
```

### mlx-audio indextts2
```python
# 有 repetition_penalty，但使用 Python list 操作
sampler = make_sampler(temp=temperature, top_k=top_k)

def apply_repetition_penalty(logits, generated_tokens, penalty):
    if len(generated_tokens) > 0:
        indices = mx.array(generated_tokens)  # Python list → mx.array 每次转换
        ...
```

### 潜在优化方向

1. **使用 mlx_lm.sample_utils**:
   - `make_sampler()` 返回的采样函数已经用 `@mx.compile` 优化
   - 可能带来性能提升

2. **代码简化**:
   - 当前手写的 top_k/top_p/repetition_penalty 逻辑可替换为 mlx_lm 工具函数
   - 减少维护负担

3. **待验证**:
   - mlx-audio indextts v1.5 有报告存在问题
   - mlx-audio indextts2 未经充分测试
   - 需要实际测试性能差异

### 结论
可以考虑将采样逻辑迁移到 mlx_lm.sample_utils，但需要:
1. 保留 repetition_penalty 功能 (mlx_lm 有 `make_logits_processors`)
2. 实际测试确认音质和性能

---

## 相关文档
- [indextts_1.5_alignment.md](indextts_1.5_alignment.md) - 1.5 踩坑经验
- [indextts_2.0_alignment.md](indextts_2.0_alignment.md) - 2.0 踩坑经验
