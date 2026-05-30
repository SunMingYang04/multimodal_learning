# Day 01 - 学习记录

## 今日目标

今天的核心目标是完成 Transformer 与 Self-Attention 的入门学习，并通过 PyTorch 手写 scaled dot-product attention，建立对 Q/K/V、attention score matrix、softmax 权重分布和输出聚合过程的基本理解。

今天不追求完整掌握 Transformer，而是重点理解以下问题：

1. Q、K、V 分别在 attention 中承担什么作用；
2. 为什么 `Q @ K^T` 会得到 token-to-token 关系矩阵；
3. 为什么 softmax 后每一行权重和为 1；
4. 为什么最后需要用 `attention_weights @ V` 得到新的 token 表示。

---

## 今日完成

### 理论学习

- 初步理解 Transformer 中 attention 的基本计算流程。
- 理解 token 表示进入模型后，不是直接处理文字本身，而是处理 embedding vector。
- 理解 self-attention 的核心思想：每个 token 根据与其他 token 的相关性，从其他 token 中聚合信息。
- 初步区分 Q、K、V 的功能：Q/K 用于计算相关性，V 用于提供被聚合的内容。

### 代码实现

已完成 `models/attention.py` 中的 scaled dot-product attention 实现。

核心代码逻辑：

```python
scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
attention_weights = F.softmax(scores, dim=-1)
outputs = torch.matmul(attention_weights, V)
```

代码中已经包含：

- Q/K/V 输入说明；
- attention scores 计算；
- softmax 权重归一化；
- attention output 计算；
- attention matrix 打印；
- row sums 检查。

---

## 核心理解

### 1. Attention 的一句话理解

Attention 的本质是：

> 先判断当前 token 应该关注谁，再从被关注对象那里聚合信息。

更具体地说：

- `Q @ K^T` 用来计算 token 与 token 之间的相关性；
- `softmax` 把相关性转换成注意力分布；
- `V` 提供真正被聚合的内容信息；
- `attention_weights @ V` 根据注意力权重从 V 中加权汇总信息，得到新的 token 表示。

### 2. Q / K / V 的作用

| 变量 | 直观理解 | 作用 |
|---|---|---|
| Q | 我想找什么 | 发起查询，用来和 K 计算相似度 |
| K | 我有什么标签 / 能不能被匹配 | 被 Q 匹配，用来计算 attention scores |
| V | 真正被取走的内容 | 提供最终被聚合的信息 |
| attention weights | 每个内容该取多少 | 决定从不同 V 中聚合多少信息 |

### 3. Shape 理解

假设：

```python
Q.shape = [2, 4, 8]
K.shape = [2, 4, 8]
V.shape = [2, 4, 8]
```

其中：

- `2` 表示 batch size；
- `4` 表示 sequence length，也就是每个样本有 4 个 token；
- `8` 表示 embedding dimension。

计算：

```python
scores = Q @ K.transpose(-1, -2)
```

得到：

```python
scores.shape = [2, 4, 4]
```

原因是：每个样本中，Q 有 4 个 query token，K 有 4 个 key token。每一个 query token 都需要分别和所有 4 个 key token 计算相似度，因此单个样本得到一个 `4 × 4` 的 token-to-token attention score matrix。由于 batch size 为 2，所以最终 shape 是 `[2, 4, 4]`。

---

## 代码产出

今日主要产出文件：

```text
models/attention.py
```

当前实现已经达到 Day01 最低要求：

- [x] 实现 scaled dot-product attention；
- [x] 打印 attention matrix；
- [x] 检查 softmax 后 row sums 是否接近 1；
- [x] 理解输入输出 tensor shape；
- [x] 能解释 `Q @ K^T`、`softmax`、`attention_weights @ V` 的作用。

### 代码评价

当前代码完成度良好。主公式实现正确，并且通过 row sums 验证了 softmax 后每一行确实形成概率分布。

需要注意的是，当前测试代码中使用：

```python
q = k = v = torch.randn(2, 4, 8)
```

这适合演示 self-attention，但 Q/K/V 指向同一个 Tensor。后续应补充一组独立 Q/K/V 测试：

```python
q = torch.randn(2, 4, 8)
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 8)
```

这样可以更好地为后续 cross-attention、多模态对齐和 Q-Former 做准备。

---

## 今日考校

### Q1：为什么 token 和 token 的关系矩阵是 4×4？

我的原始回答：

> 是因为一个 query 的一个 token 对应 key 的四个 token。

修正后的完整理解：

> 因为 Q 中有 4 个 query token，K 中也有 4 个 key token。每一个 query token 都需要分别和所有 4 个 key token 计算相似度，因此单个样本会得到一个 `4 × 4` 的 token-to-token attention score matrix。若 batch size 为 2，则最终 scores 的 shape 为 `[2, 4, 4]`。

### Q2：为什么 softmax 后每一行和为 1？

我的原始回答：

> 因为 softmax 把权重参数按照不同 token 进行了分配。

修正后的完整理解：

> softmax 会把某一个 query token 对所有 key token 的 attention scores 转换成一个概率分布。这个分布表示当前 query token 应该从不同 value token 中聚合多少信息。由于它是概率分布，所以每一行的权重和为 1。

注意：attention weights 不是固定模型参数，而是由当前输入的 Q 和 K 动态计算出来的权重分布。

### Q3：为什么最后要用 `attention_weights @ V`？

我的原始回答：

> V 在这里提供的是关注的内容。

修正后的完整理解：

> V 提供的是可以被聚合的内容信息。Q 和 K 负责计算当前 token 应该关注哪些 token，而 `attention_weights @ V` 则根据这些注意力权重，从对应的 V 中加权取出信息，得到新的 token 表示。

---

## 遇到的问题

1. 对 `Q @ K^T` 后为什么是 `[batch, seq_len, seq_len]` 的理解一开始还不够完整。现在已经明确：这是因为每个 query token 都要和所有 key token 建立关系。
2. 对 attention weights 的性质需要注意：它不是固定参数，而是由当前输入动态计算出的概率分布。
3. 当前只实现了 single-head attention，还没有进入 multi-head attention 中的 head 拆分、reshape 和 transpose。

---

## 明日计划

明天进入 Multi-Head Attention。

### 理论目标

需要回答核心问题：

> 既然 single-head attention 已经能让 token 关注其他 token，为什么还需要 multi-head attention？

重点理解：

- 不同 head 可以学习不同类型的 token 关系；
- `embed_dim = num_heads × head_dim`；
- 为什么要把 `[batch, seq_len, embed_dim]` 变成 `[batch, num_heads, seq_len, head_dim]`。

### 代码目标

创建：

```text
models/multi_head_attention.py
```

最低要求：

- 使用 `nn.Linear` 生成 Q/K/V；
- 实现 head 拆分；
- 打印每一步 tensor shape；
- 输出 shape 保持为 `[batch, seq_len, embed_dim]`；
- 对比 single-head attention 与 multi-head attention 的区别。

---

## 今日总结

今天完成了从“看懂 attention 公式”到“能够手写 attention 并解释 Q/K/V 作用”的关键转换。

今天最重要的理解是：

> Attention = 先判断该关注谁，再从被关注对象那里取信息。

这是后续学习 ViT、CLIP、BLIP、LLaVA、Qwen-VL 和 InternVL 的共同基础。
