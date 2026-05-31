# Day 02 - 学习记录

## 今日目标

今天的目标是从 Day01 的 single-head scaled dot-product attention 进一步扩展到 Multi-Head Attention，重点理解为什么需要多个 head，以及如何通过 `view / transpose / contiguous / view` 完成多头拆分与合并。

今日核心问题：

1. 为什么 single-head attention 不够？
2. 为什么 multi-head attention 能增强表达能力？
3. `embed_dim`、`num_heads`、`head_dim` 三者是什么关系？
4. 为什么要把 `[batch, seq_len, embed_dim]` 变成 `[batch, num_heads, seq_len, head_dim]`？
5. 为什么最后还要把多个 head concat 回 `[batch, seq_len, embed_dim]`？

---

## 今日完成

### 理论理解

完成了 Day02 的前置考校问题：

> 为什么 single-head attention 不够？为什么还需要 multi-head attention？

我的原始回答：

> 因为多头注意力可能会获得更多特征。

修正后的完整理解：

> Single-head attention 只有一个 attention 分布，因此只能从一个角度建模 token 之间的关系。Multi-head attention 会把 embedding 维度拆分成多个 head，每个 head 在不同的表示子空间中独立计算 attention。这样，不同 head 可以学习不同类型的关系，例如语义关系、位置关系、局部关系和长距离依赖。最后再将多个 head 的输出拼接起来，使模型能够从多个视角综合表示 token 信息。

今日需要记住的一句话：

> Multi-head attention = 多个子空间里的多种关系建模。

### 代码实现

已完成 `models/multi_head_attention.py` 的初版实现。

当前代码已经包含：

- `MultiHeadAttention(nn.Module)` 类；
- `W_q / W_k / W_v / W_o` 四个线性层；
- `embed_dim % num_heads == 0` 的合法性检查；
- `head_dim = embed_dim // num_heads`；
- Q/K/V 的线性投影；
- 使用 `view(...).transpose(1, 2)` 完成 head split；
- 调用 Day01 的 `attention(Q, K, V)` 函数完成每个 head 内部的 scaled dot-product attention；
- 使用 `transpose(1, 2).contiguous().view(...)` 合并多头；
- 输出维度保持为 `[batch_size, seq_len, embed_dim]`；
- attention weights 的 row sums 检查。

当前测试配置：

```python
x = torch.randn(2, 6, 64)
mha = MultiHeadAttention(embed_dim=64, num_heads=4)
out, weights = mha(x, x, x)
```

对应 shape：

```text
Input: [2, 6, 64]
num_heads: 4
head_dim: 16
After split: [2, 4, 6, 16]
Attention weights: [2, 4, 6, 6]
Output: [2, 6, 64]
```

---

## 核心理解

### 1. 为什么需要 Multi-Head Attention？

Single-head attention 只有一个注意力分布，因此它只能从一个角度建模 token 之间的关系。

Multi-head attention 将 `embed_dim` 拆分成多个 head，每个 head 在自己的子空间中独立计算 attention。这样可以让不同 head 学习不同类型的关系。

例如：

- 一个 head 可能关注语义相关 token；
- 一个 head 可能关注位置邻近 token；
- 一个 head 可能关注长距离依赖；
- 一个 head 可能关注句法结构或局部模式。

最后把多个 head 的输出 concat 起来，就得到更丰富的 token 表示。

### 2. `embed_dim`、`num_heads`、`head_dim` 的关系

Multi-head attention 中：

```text
embed_dim = num_heads × head_dim
```

当前代码中：

```text
embed_dim = 64
num_heads = 4
head_dim = 16
```

因此：

```text
64 = 4 × 16
```

### 3. Head split 的 shape 变化

原始输入：

```text
[batch_size, seq_len, embed_dim]
```

以当前代码为例：

```text
[2, 6, 64]
```

经过线性层后，Q/K/V shape 仍然是：

```text
[2, 6, 64]
```

然后使用：

```python
.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
```

先变成：

```text
[2, 6, 4, 16]
```

再转置为：

```text
[2, 4, 6, 16]
```

这样每个 head 可以独立计算自己的 attention。

### 4. 为什么 attention weights 是 `[2, 4, 6, 6]`？

因为当前有：

- batch size = 2；
- num_heads = 4；
- seq_len = 6。

每个 batch、每个 head 内部，6 个 query token 都要和 6 个 key token 计算相关性，因此每个 head 的 attention matrix 是 `6 × 6`。

所以 attention weights 的 shape 是：

```text
[batch_size, num_heads, seq_len, seq_len]
= [2, 4, 6, 6]
```

### 5. 为什么最后要 concat heads？

每个 head 只处理一部分子空间信息。当前配置中，每个 head 的维度是 16，4 个 head 拼接后重新得到 64 维表示。

```text
16 × 4 = 64
```

因此输出要从：

```text
[2, 4, 6, 16]
```

先转回：

```text
[2, 6, 4, 16]
```

再合并为：

```text
[2, 6, 64]
```

这样输出维度与输入维度一致，方便后续 Transformer block 中进行残差连接。

---

## 代码产出

今日主要产出文件：

```text
models/multi_head_attention.py
```

当前核心代码结构：

```python
Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
K = self.W_k(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
V = self.W_v(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

x, attn = attention(Q, K, V)

x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
output = self.W_o(x)
```

今日最低交付物完成情况：

- [x] 创建 `models/multi_head_attention.py`
- [x] 实现 Multi-Head Attention 类
- [x] 使用 `nn.Linear` 生成 Q/K/V
- [x] 实现 head split
- [x] 调用 scaled dot-product attention
- [x] 实现 head concat
- [x] 保持最终输出 shape 为 `[batch, seq_len, embed_dim]`
- [x] 添加 row sums 检查
- [ ] 进一步独立解释 `contiguous()` 的作用
- [ ] 增加更详细的每一步 shape 打印

---

## 遇到的问题

### 1. 理论表达仍需更精确

原始回答中“多头注意力可能会获得更多特征”方向正确，但不够准确。更好的表达应该是：

> 多头注意力让模型在多个表示子空间中分别建模不同类型的 token 关系。

### 2. `transpose(1, 2)` 需要反复理解

`view` 后的 shape 是：

```text
[batch_size, seq_len, num_heads, head_dim]
```

但为了让每个 head 独立计算 attention，需要变成：

```text
[batch_size, num_heads, seq_len, head_dim]
```

因此必须使用 `transpose(1, 2)`。

### 3. `contiguous()` 的作用还需要继续学习

在 `transpose` 后，Tensor 的内存布局可能不连续。后续再使用 `view` 前，需要调用 `.contiguous()`，否则可能出现 view 报错或行为不符合预期。

### 4. import 路径可进一步改进

当前代码使用：

```python
from attention import attention
```

在某些运行方式下可能出现路径问题。后续可以考虑改为更稳妥的包结构，或在同目录运行时明确执行方式。

---

## 跨对话同步记录

本日志已同步此前其他对话中确定的长期学习规则与仓库定位，避免 Day02 只记录单日代码，而脱离三个月多模态算法实习冲刺主线。

### 1. 仓库定位

当前 GitHub 仓库不是普通学习笔记，而是用于积累：

- 公开学习档案；
- 项目证据链；
- 面试素材库；
- 可复现实验记录；
- 后续简历 bullet 的原始材料。

因此每天日志必须服务于最终目标：三个月内形成多模态算法实习候选人的可验证能力。

### 2. 当前阶段定位

当前处于 Week 01 的基础补齐阶段，主线是：

```text
Transformer / Attention → Multi-Head Attention → Positional Encoding → ViT Patch Embedding → CLIP 准备
```

Day01 已完成 scaled dot-product attention；Day02 推进到 multi-head attention。当前不追求快速堆模型名，而是先把 Transformer 内部的 tensor shape 和计算流打牢。

### 3. 与三个月路线的关系

Day02 的 Multi-Head Attention 是后续以下内容的基础：

- ViT：图像 patch token 之间通过 self-attention 建模关系；
- CLIP：图像编码器和文本编码器都依赖 Transformer/Attention 表征；
- BLIP / BLIP-2：理解图文交互和 Q-Former 前必须理解 attention；
- LLaVA / Qwen-VL / InternVL：视觉 token、文本 token、projector 与 LLM 对接都需要 shape 意识；
- LoRA / QLoRA：后续选择 target modules 时也需要理解 attention projection 层。

### 4. 每日动态推进规则

后续每日不再机械执行固定计划，而采用动态导师制流程：

```text
当日完成 → GitHub 检查 → 概念考校 → 日志更新 → 第二日计划生成
```

第二日任务会根据当天掌握情况分为三类：

- A 类：代码运行成功，shape 能解释清楚，核心问题回答稳定，则正常进入下一主题；
- B 类：代码基本完成，但 shape 或概念表达不稳，则第二天先补薄弱点，再进入新主题；
- C 类：代码未跑通或核心概念混乱，则不推进新内容，继续重写与修正当天任务。

### 5. Day02 当前判定

Day02 当前状态暂定为 B 类偏上：

- 已完成 multi-head attention 初版代码；
- 已理解“多个子空间里的多种关系建模”这一核心思想；
- 仍需巩固 `view / transpose / contiguous / concat heads` 的 shape 变化；
- 还需要补充更详细的 shape 打印和更稳妥的 import 方式。

因此 Day03 不直接大幅推进新内容，而是采用：

```text
上午巩固 Multi-Head Attention shape
下午视情况进入 Positional Encoding
```

---

## 明日计划

明天不急着进入全新的模型主题，先进行 Day02 巩固，再根据掌握情况进入 Positional Encoding。

### 上午：Multi-Head Attention 巩固

目标：彻底理解 shape 变化。

需要能不看代码说出：

```text
[2, 6, 64]
→ [2, 6, 4, 16]
→ [2, 4, 6, 16]
→ [2, 4, 6, 6]
→ [2, 4, 6, 16]
→ [2, 6, 4, 16]
→ [2, 6, 64]
```

需要补充：

- 更详细打印 Q/K/V 每一步 shape；
- 手动写一段注释解释 `view`、`transpose`、`contiguous`；
- 尝试使用更小配置 `[2, 4, 8]`、`num_heads=2` 重新跑一遍。

### 下午：Positional Encoding 入门

如果上午 Multi-Head Attention 巩固顺利，下午开始理解 Transformer 为什么需要位置信息。

核心问题：

> 如果没有 positional encoding，Transformer 是否知道 token 的顺序？

最低目标：

- 理解 attention 本身对顺序不敏感；
- 理解 positional encoding 的作用是给 token 注入位置信息；
- 创建 `notes/transformer/positional_encoding.md` 初版笔记。

---

## 今日总结

今天已经从 single-head attention 推进到 multi-head attention，并完成了初版代码实现。当前最重要的收获是理解：

> Multi-head attention 不是简单获得更多特征，而是在多个子空间里分别建模不同类型的 token 关系。

结合此前确定的三个月冲刺目标，Day02 的意义不只是完成一个 attention 模块，而是为后续 ViT、CLIP、BLIP、LLaVA、Qwen-VL 和 InternVL 打下 shape 与模块理解基础。

今晚到此为止，不继续推进新内容。明天先巩固 shape，再视情况进入 positional encoding。
