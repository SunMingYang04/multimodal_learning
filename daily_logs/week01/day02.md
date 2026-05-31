下面这一版可以直接粘贴到：

`daily_logs/week01/day02.md`

---

# Day 02 - Multi-Head Attention

## 今日目标

今天的核心目标是从 Single-Head Attention 扩展到 Multi-Head Attention，理解 Transformer 为什么要把 embedding 拆分为多个 head 并行计算 attention。

今天重点解决以下问题：

1. 为什么 single-head attention 不够；
2. 为什么 `embed_dim = num_heads × head_dim`；
3. 为什么要把 `[batch, seq_len, embed_dim]` 变换为 `[batch, num_heads, seq_len, head_dim]`；
4. 为什么多个 head 最后要 concat 回 `embed_dim`；
5. 为什么 concat 后还需要一个输出线性层 `W_o`。

---

## 今日完成

### 1. 理论学习

今天完成了 Multi-Head Attention 的基本概念学习，并重点理解了以下内容：

* Single-head attention 只在一个表示子空间中计算 token 关系，表达能力有限；
* Multi-head attention 将 embedding 拆分成多个 head，使不同 head 可以学习不同类型的 token-to-token 关系；
* `embed_dim` 必须能被 `num_heads` 整除，因为每个 head 需要分到相同维度的子空间；
* 每个 head 内部都会独立计算一张 attention matrix；
* 多个 head 的输出需要 concat 回原始 embedding 维度；
* `W_o` 的作用不是单纯保持输出 shape，而是对多个 head 拼接后的信息进行线性融合和重新投影。

### 2. 代码实现

今日完成了 `models/multi_head_attention.py` 的编写和小修。

当前代码中包含：

* `MultiHeadAttention` 类；
* `W_q`、`W_k`、`W_v`、`W_o` 四个线性层；
* `embed_dim % num_heads == 0` 断言；
* Q/K/V 的多头拆分；
* `transpose(1, 2)` 维度交换；
* 调用 Day01 实现的 scaled dot-product attention；
* concat heads；
* output projection；
* attention row sums 检查。

---

## 核心理解

### 1. 为什么需要 Multi-Head Attention？

Single-head attention 只在一个表示子空间中计算 token 之间的关系，表达能力有限。

Multi-head attention 将 embedding 拆分成多个 head，使不同 head 可以在不同子空间中学习不同类型的 token-to-token 关系，因此能够捕捉更丰富的语义、结构或位置关系。

一句话理解：

> 多个 head = 多个关系视角。

---

### 2. `embed_dim`、`num_heads`、`head_dim` 的关系

Multi-head attention 会把原始 embedding 维度平均拆分给多个 head。

因此：

```text
embed_dim = num_heads × head_dim
```

例如：

```text
embed_dim = 64
num_heads = 4
head_dim = 16
```

也就是：

```text
64 = 4 × 16
```

所以 `embed_dim` 必须能被 `num_heads` 整除，否则无法平均拆分。

---

### 3. Shape 链条理解

以今日代码中的测试输入为例：

```python
x.shape = [2, 6, 64]
num_heads = 4
head_dim = 16
```

完整 shape 变化为：

```text
输入 x: [2, 6, 64]
Linear 后 Q/K/V: [2, 6, 64]
view 后: [2, 6, 4, 16]
transpose 后: [2, 4, 6, 16]
attention scores: [2, 4, 6, 6]
attention weights: [2, 4, 6, 6]
head output: [2, 4, 6, 16]
transpose back: [2, 6, 4, 16]
concat heads: [2, 6, 64]
final output: [2, 6, 64]
```

其中：

```text
[2, 4, 6, 16]
```

表示：

```text
2 个样本，每个样本有 4 个 head，每个 head 中有 6 个 token，每个 token 是 16 维表示。
```

---

### 4. 为什么要 `transpose(1, 2)`？

`view` 后的 shape 是：

```text
[batch, seq_len, num_heads, head_dim]
```

但 attention 需要让每个 head 独立计算 token-to-token 关系，因此要把 `num_heads` 维度提前：

```text
[batch, num_heads, seq_len, head_dim]
```

这样每个 head 内部都可以计算：

```text
[seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
```

最终每个样本的每个 head 都有一张独立的 attention matrix。

---

### 5. 为什么除以 `sqrt(head_dim)`？

在 multi-head attention 中，点积 attention 是在每个 head 的低维子空间中进行的。

每个 head 中 Q/K 的最后一维是 `head_dim`，因此缩放因子应该是：

```text
sqrt(head_dim)
```

而不是：

```text
sqrt(embed_dim)
```

因为每个 head 的 attention score 是由该 head 内部的 `head_dim` 维向量点积得到的。

---

### 6. 为什么 softmax 使用 `dim=-1`？

attention scores 的 shape 是：

```text
[batch, num_heads, seq_q, seq_k]
```

最后一维 `seq_k` 表示：

```text
某一个 query token 对所有 key token 的相关性分数。
```

因此 softmax 要在最后一维进行，使每个 query token 对所有 key token 的权重和为 1。

这意味着每一行 attention weights 都是一组概率分布。

---

### 7. 为什么 concat heads？

每个 head 只输出 `head_dim` 维的子空间信息。

例如：

```text
num_heads = 4
head_dim = 16
```

每个 token 在每个 head 中只得到 16 维表示。4 个 head 的输出需要重新拼接：

```text
16 × 4 = 64
```

从而恢复完整的 `embed_dim` 维 token 表示。

一句话理解：

> 拆 head 是为了分开看，concat 是为了合起来用。

---

### 8. 为什么最后还需要 `W_o`？

`W_o` 的作用不只是保证输出 shape 与输入一致。

concat heads 后，不同 head 的信息只是被简单拼接在一起。`W_o` 会对拼接后的多头信息进行一次线性变换，使不同 head 的信息进一步融合，并映射回统一的 `embed_dim` 表示空间。

因此，`W_o` 的更准确理解是：

> 对多个 head 拼接后的信息进行线性融合和重新投影。

---

## 代码产出

今日主要产出文件：

```text
models/multi_head_attention.py
```

当前实现已经达到 Day02 最低要求：

* [x] 实现 Multi-Head Attention 类；
* [x] 使用线性层生成 Q/K/V；
* [x] 正确计算 `head_dim`；
* [x] 完成 head 拆分；
* [x] 使用 `transpose(1, 2)` 调整维度；
* [x] 调用 scaled dot-product attention；
* [x] concat heads；
* [x] 使用 output projection；
* [x] 打印关键中间 shape；
* [x] 检查 attention row sums 是否接近 1。

---

## 实验结果

今日测试代码使用：

```python
x = torch.randn(2, 6, 64)
mha = MultiHeadAttention(embed_dim=64, num_heads=4)
out, weights = mha(x, x, x)
```

输出满足：

```text
Input: torch.Size([2, 6, 64])
Output: torch.Size([2, 6, 64])
Attn weights shape: torch.Size([2, 4, 6, 6])
```

其中：

```python
weights.sum(dim=-1)
```

用于检查每个 query token 对所有 key token 的 attention 权重和是否为 1。

这说明 softmax 后每一行 attention weights 都是合法的概率分布。

---

## 今日考校

### Q1：为什么 single-head attention 不够？为什么需要 multi-head attention？

我的回答：

> 是为了获得 token 对于多个 token 不同关系的特征信息。

评价：基本正确，但需要更精确。

修正后的完整理解：

> Single-head attention 只在一个表示子空间中计算 token 关系，表达能力有限。Multi-head attention 将 embedding 拆成多个 head，使不同 head 可以在不同子空间中学习不同类型的 token-to-token 关系，因此可以捕捉更丰富的语义、结构或位置关系。

得分：80 / 100

---

### Q2：`x.shape = [2, 6, 64]`，`num_heads = 4` 时，各 shape 是什么？

我的回答：

> 16；[2, 4, 6, 16]；[2, 6, 64]

评价：`head_dim`、Q/K/V 拆分后的 shape、最终 output shape 正确，但漏掉了 attention weights 的 shape。

修正后的完整答案：

```text
head_dim = 16
Q/K/V 经过 view + transpose 后: [2, 4, 6, 16]
attention weights: [2, 4, 6, 6]
最终 output: [2, 6, 64]
```

得分：75 / 100

---

### Q3：为什么 `embed_dim` 必须能被 `num_heads` 整除？

我的回答：

> 为了保证输入维度能够被均分。

评价：正确。

修正后的完整理解：

> 因为 multi-head attention 会把 `embed_dim` 平均拆分给多个 head，每个 head 的维度为 `head_dim = embed_dim / num_heads`。如果不能整除，就无法将 embedding 均分到多个 head 中。

得分：90 / 100

---

### Q4：解释 `self.W_q(q).view(...).transpose(1, 2)`。

我的回答：

> 将查询队列通过全连接网络输入；将输入维度根据头数量均分；将 head 维度与 token 维度互换。

评价：基本正确。“查询队列”建议改为“query 输入张量”或“输入 token 表示”。

修正后的完整理解：

* `self.W_q(q)`：将输入 token 表示线性映射为 Query 表示；
* `view(batch_size, -1, num_heads, head_dim)`：把原始 `embed_dim` 拆分为 `num_heads × head_dim`；
* `transpose(1, 2)`：交换 `seq_len` 和 `num_heads` 维度，使每个 head 可以独立计算 attention。

得分：85 / 100

---

### Q5：为什么除以 `sqrt(head_dim)`，而不是 `sqrt(embed_dim)`？

我的回答：

> 因为是在各个头的低维子空间内进行的运算，只与当前维度有关。

评价：正确。

修正后的完整理解：

> 每个 head 内部的 Q/K 点积是在 `head_dim` 维空间中进行的，因此 score 的尺度由 `head_dim` 决定，而不是由完整的 `embed_dim` 决定。

得分：95 / 100

---

### Q6：为什么 softmax 写成 `dim=-1`？最后一维代表什么？

我的回答：

> 表示行，也就是一个查询 token 对于多个键 token 的关系。

评价：方向正确，但表述需要修正。`dim=-1` 不是“表示行”，而是在最后一维，也就是所有 key token 上做 softmax。

修正后的完整理解：

> attention scores 的 shape 是 `[batch, num_heads, seq_q, seq_k]`。最后一维 `seq_k` 表示某个 query token 对所有 key token 的分数，因此 softmax 应该在最后一维进行，使每个 query token 对所有 key token 的权重和为 1。

得分：75 / 100

---

### Q7：解释 `transpose(1, 2).contiguous().view(...)`。

我的回答：

> 将维度换回；保证内存连续；将多个 head 重新合并为一个。

评价：正确。

修正后的完整理解：

* `transpose(1, 2)`：把 `[batch, num_heads, seq_len, head_dim]` 换回 `[batch, seq_len, num_heads, head_dim]`；
* `contiguous()`：保证张量在内存中连续，便于安全使用 `view`；
* `view(batch_size, -1, embed_dim)`：将 `num_heads × head_dim` 合并回 `embed_dim`。

得分：95 / 100

---

### Q8：为什么最后还要经过输出线性层 `W_o`？

我的回答：

> 保证输出与输入形状相同。

评价：部分正确，但不完整。输出形状在 concat 后已经回到 `[batch, seq_len, embed_dim]`，`W_o` 更重要的作用是融合多个 head 的信息并重新投影。

修正后的完整理解：

> concat heads 后，不同 head 的信息只是被拼接在一起。`W_o` 会对拼接后的多头信息进行线性变换，使不同 head 的信息进一步融合，并映射回统一的 `embed_dim` 表示空间。

得分：60 / 100

---

## 考校总评

今日考校总体通过。

```text
Q1: 80 / 100
Q2: 75 / 100
Q3: 90 / 100
Q4: 85 / 100
Q5: 95 / 100
Q6: 75 / 100
Q7: 95 / 100
Q8: 60 / 100

平均分: 81.9 / 100
```

### 当前掌握情况

已经掌握：

* multi-head attention 的基本动机；
* `embed_dim`、`num_heads`、`head_dim` 的关系；
* Q/K/V 的多头拆分；
* `transpose(1, 2)` 的作用；
* `sqrt(head_dim)` 缩放原因；
* concat heads 的基本逻辑；
* attention row sums 的检查方法。

仍需巩固：

* attention weights 的完整 shape：`[batch, num_heads, seq_q, seq_k]`；
* `dim=-1` 的精确含义：在所有 key token 上做 softmax；
* `W_o` 的作用不是单纯保持 shape，而是融合多头信息并重新投影。

---

## 今日总结

今天完成了 Multi-Head Attention 的理论理解、代码实现和考校验证。

最重要的理解是：

> Multi-head attention 先把 embedding 拆成多个 head，让不同 head 在不同子空间中学习不同 token 关系；然后再把多个 head 的输出拼接回完整 embedding，并通过 `W_o` 进行融合和重新投影。

今日核心 shape 链条：

```text
[2, 6, 64]
→ [2, 6, 4, 16]
→ [2, 4, 6, 16]
→ [2, 4, 6, 6]
→ [2, 4, 6, 16]
→ [2, 6, 4, 16]
→ [2, 6, 64]
```

---

## 遇到的问题

1. 一开始对 Multi-Head Attention 的理解停留在“获取多个特征关系”层面，后来进一步明确为：不同 head 在不同子空间中学习不同 token-to-token 关系。
2. 在 shape 推导中漏掉了 attention weights 的 shape，后续需要固定记忆：`[batch, num_heads, seq_q, seq_k]`。
3. 对 `softmax(dim=-1)` 的表述不够精确，后续需要明确：最后一维是所有 key token。
4. 对 `W_o` 的理解最初偏向“保证输出形状一致”，后来修正为：对多个 head 拼接后的信息进行融合和重新投影。

---

## 明日计划

明天进入 Transformer Encoder Block。

重点理解：

1. Multi-Head Attention 在 Encoder Block 中的位置；
2. 残差连接为什么重要；
3. LayerNorm 的作用；
4. Feed Forward Network 为什么是逐 token 处理；
5. 一个 Encoder Block 的完整数据流：

```text
x
→ Multi-Head Attention
→ Add & Norm
→ Feed Forward Network
→ Add & Norm
→ output
```

代码目标：

```text
models/transformer_encoder_block.py
```

最低要求：

* 调用今天实现的 `MultiHeadAttention`；
* 加入 residual connection；
* 加入 `nn.LayerNorm`；
* 加入简单 FFN；
* 输入输出 shape 保持 `[batch, seq_len, embed_dim]`。
