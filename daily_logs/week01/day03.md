# Day 03 - Transformer Encoder Block

## 今日目标

今天的核心目标是从 Multi-Head Attention 进一步扩展到完整的 Transformer Encoder Block，理解 Encoder Block 中 Multi-Head Self-Attention、Residual Connection、LayerNorm 和 Feed Forward Network 的结构关系。

今天重点解决以下问题：

1. Encoder Block 为什么需要同时包含 Multi-Head Attention 和 FFN；
2. Residual Connection 的作用是什么；
3. LayerNorm 为什么放在残差连接之后；
4. FFN 为什么采用 `embed_dim → ff_dim → embed_dim`；
5. Encoder Block 为什么必须保持输入输出 shape 一致。

---

## 今日完成

### 1. 理论学习

今天完成了 Transformer Encoder Block 的基本结构学习，重点理解了以下内容：

- Multi-Head Attention 负责 token 与 token 之间的信息交互；
- FFN 不负责 token 之间通信，而是对每个 token 自身进行非线性特征变换；
- Residual Connection 用于保留原始输入信息，并帮助深层网络稳定训练；
- LayerNorm 用于稳定每个 token 的 embedding 维度上的数值分布；
- Encoder Block 的输入输出 shape 必须一致，这样 residual connection 才能成立，也便于多个 Encoder Block 连续堆叠。

### 2. 代码实现

今日完成了 `models/transformer_encoder_block.py` 的编写和运行验证。

当前代码实现了：

- `TransformerEncoderBlock` 类；
- 调用 Day02 实现的 `MultiHeadAttention`；
- 两个 `LayerNorm` 层；
- 一个 FFN 模块；
- 两次 Residual Connection；
- Post-LN 结构：`LayerNorm(x + sublayer(x))`；
- attention weights 输出；
- shape 打印与 row sums 检查。

### 3. 运行验证

测试配置：

```python
batch_size = 2
seq_len = 6
embed_dim = 64
num_heads = 4
ff_dim = 256
```

核心 shape 流程：

```text
Input: [2, 6, 64]
MHA output: [2, 6, 64]
After first residual + norm: [2, 6, 64]
FFN output: [2, 6, 64]
Final output: [2, 6, 64]
Attention weights: [2, 4, 6, 6]
```

运行结果显示，Encoder Block 的输入输出 shape 保持一致，attention weights 的 row sums 接近 1，说明 attention softmax 仍然正常工作。

---

## 核心理解

### 1. Encoder Block 的整体结构

Transformer Encoder Block 可以理解为两个子层的组合：

```text
Input x
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection + LayerNorm
  ↓
Feed Forward Network
  ↓
Residual Connection + LayerNorm
  ↓
Output
```

对应代码结构：

```python
attn_output, attn_weights = self.self_attn(x, x, x)
x = self.norm1(x + attn_output)

ffn_output = self.ffn(x)
x = self.norm2(x + ffn_output)
```

这说明 Encoder Block 并不是只有 attention，而是由 attention、残差连接、归一化和前馈网络共同组成。

### 2. Multi-Head Attention 的作用

Multi-Head Attention 负责让不同 token 之间发生信息交互。

对于输入：

```text
[batch_size, seq_len, embed_dim]
```

attention 会计算 token-to-token 的关系矩阵：

```text
[batch_size, num_heads, seq_len, seq_len]
```

它回答的问题是：

> 当前 token 应该从其他 token 中关注并聚合多少信息？

在 Encoder Block 中，Self-Attention 使用：

```python
self.self_attn(x, x, x)
```

也就是说，Q、K、V 都来自同一个输入序列，因此这是 self-attention。

### 3. FFN 的作用

FFN 的核心作用不是让 token 之间通信，而是对每个 token 的表示进行非线性变换。

它通常采用：

```text
embed_dim → ff_dim → embed_dim
```

例如当前代码中：

```text
64 → 256 → 64
```

原因是：

1. 先升维到更高维空间，增强非线性表达能力；
2. 经过 ReLU 激活，引入非线性；
3. 再降回 `embed_dim`，保证输出 shape 与输入一致。

对应代码：

```python
self.ffn = nn.Sequential(
    nn.Linear(embed_dim, ff_dim),
    nn.ReLU(),
    nn.Linear(ff_dim, embed_dim)
)
```

需要注意：FFN 是 position-wise 的。它对每个 token 独立执行同一组 MLP，不会直接混合不同 token 的信息。

### 4. Residual Connection 的作用

Residual Connection 的核心形式是：

```python
x + sublayer(x)
```

它的作用包括：

- 保留原始输入信息；
- 让模型学习输入的增量变化，而不是完全重写表示；
- 缓解深层网络训练中的梯度消失问题；
- 保证多个 Encoder Block 可以稳定堆叠。

在 Encoder Block 中有两次残差连接：

```python
x = self.norm1(x + attn_output)
x = self.norm2(x + ffn_output)
```

第一处残差连接发生在 attention 子层后，第二处残差连接发生在 FFN 子层后。

### 5. LayerNorm 的作用

LayerNorm 用于稳定每个 token 的 embedding 维度上的数值分布。

对于输入：

```text
[batch_size, seq_len, embed_dim]
```

LayerNorm 通常在最后一个维度 `embed_dim` 上做归一化。

它的作用是：

- 稳定不同层之间的激活分布；
- 提升训练稳定性；
- 避免数值过大或过小影响后续计算。

当前实现采用的是 Post-LN：

```python
x = self.norm1(x + attn_output)
x = self.norm2(x + ffn_output)
```

即先做残差相加，再做 LayerNorm。

### 6. 为什么 Encoder Block 输入输出 shape 必须一致？

Encoder Block 的输入输出 shape 必须保持一致，主要有两个原因：

#### 原因一：Residual Connection 要求 shape 一致

残差连接需要做：

```python
x + sublayer(x)
```

因此 `x` 和 `sublayer(x)` 必须具有相同 shape。

例如：

```text
x:           [2, 6, 64]
attn_output: [2, 6, 64]
```

二者 shape 一致，才能相加。

#### 原因二：多个 Encoder Block 要连续堆叠

Transformer 通常会堆叠多个 Encoder Block。如果每一层输出 shape 改变，下一层就无法直接接收上一层输出。

因此，每个 Encoder Block 通常都保持：

```text
[batch_size, seq_len, embed_dim] → [batch_size, seq_len, embed_dim]
```

这样才能稳定堆叠成深层 Transformer。

---

## 代码产出

今日主要代码文件：

```text
models/transformer_encoder_block.py
```

核心代码结构：

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerEncoderBlock, self).__init__()

        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        attn_output, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, attn_weights
```

今日最低交付物完成情况：

- [x] 创建 `models/transformer_encoder_block.py`
- [x] 实现 `TransformerEncoderBlock`
- [x] 调用 Day02 的 `MultiHeadAttention`
- [x] 实现 Multi-Head Self-Attention 子层
- [x] 实现 FFN 子层
- [x] 实现两次 Residual Connection
- [x] 实现两次 LayerNorm
- [x] 保持输入输出 shape 一致
- [x] 打印 attention weights shape
- [x] 检查 attention weights row sums

---

## 今日考校

### Q1：Encoder Block 中 MHA 和 FFN 的分工是什么？

参考答案：

> Multi-Head Attention 负责 token 与 token 之间的信息交互，建模序列内部不同 token 的依赖关系。FFN 则对每个 token 的表示独立进行非线性变换，增强每个 token 自身的表达能力。也就是说，MHA 负责 token 间通信，FFN 负责 token 内特征变换。

### Q2：为什么 FFN 要采用 `embed_dim → ff_dim → embed_dim`？

参考答案：

> 先从 `embed_dim` 升维到 `ff_dim`，可以让模型在更高维空间中学习更丰富的非线性特征；经过激活函数后，再降回 `embed_dim`，保证输出维度与输入一致，从而可以继续进行残差连接和后续 Encoder Block 堆叠。

### Q3：Residual Connection 为什么重要？

参考答案：

> Residual Connection 可以保留原始输入信息，使模型学习的是对输入表示的增量修正，而不是完全重写表示。同时它有助于缓解深层网络训练中的梯度消失问题，使多个 Encoder Block 能够稳定堆叠。

### Q4：LayerNorm 在这里归一化的是哪个维度？

参考答案：

> 对于 `[batch_size, seq_len, embed_dim]` 的输入，LayerNorm 通常在最后一个维度 `embed_dim` 上进行归一化，也就是对每个 token 的 embedding 向量做归一化，而不是跨 batch 或跨 token 做归一化。

### Q5：为什么 Encoder Block 的输入输出 shape 必须一致？

参考答案：

> 因为残差连接需要 `x + sublayer(x)`，两者 shape 必须一致；同时 Transformer 需要堆叠多个 Encoder Block，如果每层输出 shape 不一致，下一层就无法直接接收上一层输出。因此 Encoder Block 通常保持 `[batch_size, seq_len, embed_dim] → [batch_size, seq_len, embed_dim]`。

---

## 遇到的问题

### 1. 需要继续区分 MHA 和 FFN 的功能

今天已经明确：

```text
MHA：token 间通信
FFN：token 内特征变换
```

但后续还需要结合实际代码进一步理解：FFN 虽然不直接混合 token，但它对每个 token 的表示进行同样的非线性变换，是 Transformer 表达能力的重要来源。

### 2. Post-LN 和 Pre-LN 还没有深入比较

当前实现采用 Post-LN：

```python
x = self.norm1(x + attn_output)
x = self.norm2(x + ffn_output)
```

也就是先残差相加，再做 LayerNorm。

后续需要了解 Pre-LN：

```python
x = x + self_attn(self.norm1(x))
x = x + ffn(self.norm2(x))
```

现代大模型中更常使用 Pre-LN，因为训练更稳定。

### 3. 需要进一步理解 FFN 的 position-wise 特性

FFN 对每个 token 独立执行同一个 MLP，因此它不负责 token 间交互。token 间交互主要由 attention 完成。

---

## 明日计划

明天进入 Positional Encoding 与 Transformer 输入表示。

### 理论目标

重点回答以下问题：

1. 为什么 Transformer 需要 positional encoding？
2. Self-Attention 本身是否知道 token 顺序？
3. 绝对位置编码和可学习位置编码的基本区别是什么？
4. 文本 token 的 embedding 和 position embedding 如何相加？
5. ViT 中图像 patch token 是否也需要位置编码？

### 代码目标

创建：

```text
models/positional_encoding.py
```

最低实现：

- 实现 sinusoidal positional encoding；
- 输入 shape：`[batch_size, seq_len, embed_dim]`；
- 输出 shape 保持不变；
- 打印 position encoding shape；
- 验证不同 position 的 encoding 不同；
- 验证同一 position 在不同 batch 中相同。

### 笔记目标

创建：

```text
notes/transformer/positional_encoding.md
```

笔记必须说明：

- attention 本身不包含顺序信息；
- positional encoding 的作用是向 token embedding 注入位置信息；
- 文本 Transformer 和 ViT 都需要位置信息；
- 位置编码是理解 ViT patch embedding 的前置基础。

---

## 今日总结

今天完成了从 Multi-Head Attention 到 Transformer Encoder Block 的关键跃迁。

Day01 理解了 attention 的基本计算：

```text
QK^T → softmax → attention_weights @ V
```

Day02 理解了多头机制：

```text
多个子空间里的多种关系建模
```

Day03 则进一步理解了完整 Encoder Block：

```text
MHA 负责 token 间通信，FFN 负责 token 内特征变换，Residual + LayerNorm 保证深层堆叠稳定。
```

今天的学习标志着已经从单个 attention 公式进入了完整 Transformer 模块结构。下一步将学习 positional encoding，为后续 ViT 和 CLIP 的图像/文本 token 表示打基础。
