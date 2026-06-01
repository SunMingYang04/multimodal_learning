# Day 04 - Positional Encoding

## 今日目标

今天的核心目标是理解 Transformer 为什么需要 Positional Encoding，并完成 sinusoidal positional encoding 的代码实现与可视化。

重点问题：

1. Self-Attention 本身是否知道 token 顺序？
2. Positional Encoding 如何给 token 注入位置信息？
3. 为什么 token embedding 和 positional encoding 可以直接相加？
4. Sinusoidal Positional Encoding 中 sin / cos 的作用是什么？
5. 为什么 ViT 中的 patch tokens 也需要位置信息？

---

## 今日完成

### 1. 理论学习

今天完成了 Positional Encoding 的基本理解：

- Self-Attention 本身主要计算 token 与 token 之间的相关性，不显式包含顺序信息；
- Positional Encoding 的作用是给 token embedding 注入位置信息；
- 输入给 Transformer 的表示可以理解为 `token_embedding + position_embedding`；
- Sinusoidal Positional Encoding 使用固定的 sin / cos 函数生成位置向量，不需要训练参数；
- 位置编码不会改变输入输出 shape；
- ViT 中图像 patch tokens 同样需要位置信息，否则模型不知道 patch 的空间位置。

### 2. 代码实现

今日完成：

```text
models/positional_encoding.py
```

当前实现包含：

- `PositionalEncoding(nn.Module)` 类；
- 使用 sin / cos 构造固定位置编码矩阵 `P`；
- 使用 `register_buffer('P', P)` 保存非训练参数；
- 在 forward 中截取当前序列长度对应的位置编码；
- 将位置编码加到输入 `x` 上；
- 使用 dropout 进行正则化；
- 打印输入、位置编码和输出 shape；
- 可视化不同维度的位置编码波形。

### 3. 笔记产出

今日完成：

```text
notes/transformer/positional_encoding.md
```

笔记内容包括：

- Transformer 为什么需要位置编码；
- Positional Encoding 的作用；
- shape 与 broadcasting；
- sinusoidal positional encoding；
- `register_buffer` 的作用；
- dropout 的作用；
- 与 ViT patch tokens 的关系。

---

## 核心理解

### 1. 为什么 Transformer 需要位置编码？

Self-Attention 的核心计算是：

```text
QK^T → softmax → attention_weights @ V
```

这个过程主要建模 token 之间的相关性，但不天然包含 token 的顺序信息。

如果没有位置编码，模型很难区分相同 token 在不同顺序下的不同语义。

例如：

```text
我 爱 你
你 爱 我
```

这两个句子包含相同 token，但语义不同，差异来自顺序。

### 2. Positional Encoding 的作用

Positional Encoding 的作用是：

> 让模型同时知道 token 是什么，以及 token 在哪里。

输入给 Transformer 的表示可以写成：

```text
input_embedding = token_embedding + position_embedding
```

其中：

- token embedding 表示内容；
- position embedding 表示位置。

### 3. Shape 理解

输入：

```text
x: [batch_size, seq_len, embed_dim]
```

位置编码矩阵：

```text
P: [1, max_len, embed_dim]
```

在 forward 中截取：

```python
self.P[:, :x.shape[1], :]
```

得到：

```text
[1, seq_len, embed_dim]
```

与输入相加：

```text
[batch_size, seq_len, embed_dim] + [1, seq_len, embed_dim]
```

输出仍然是：

```text
[batch_size, seq_len, embed_dim]
```

因此，位置编码只注入位置信息，不改变 tensor shape。

### 4. Sinusoidal Positional Encoding

Sinusoidal Positional Encoding 使用固定的正弦和余弦函数：

```python
P[:, :, 0::2] = torch.sin(X)
P[:, :, 1::2] = torch.cos(X)
```

其中：

- 偶数维使用 sin；
- 奇数维使用 cos；
- 不同维度对应不同频率；
- 不同 position 得到不同的位置向量。

### 5. register_buffer 的作用

代码中使用：

```python
self.register_buffer('P', P)
```

说明 `P` 不是可训练参数，但仍然属于模型状态。

它的作用是：

- 不参与梯度更新；
- 会保存进 `state_dict`；
- 会随着 `model.to(device)` 一起移动到 CPU 或 GPU。

### 6. 和 ViT 的关系

ViT 会把图像切成 patch tokens。

如果没有 position embedding，模型只知道图像中有哪些 patch，却不知道这些 patch 在图像中的空间位置。

因此，位置编码不仅对文本 Transformer 重要，也是后续学习 ViT Patch Embedding 和 CLIP Vision Encoder 的基础。

---

## 代码产出

今日主要代码文件：

```text
models/positional_encoding.py
```

核心代码：

```python
P = torch.zeros((1, max_len, num_hiddens))
X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
    10000,
    torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
)
P[:, :, 0::2] = torch.sin(X)
P[:, :, 1::2] = torch.cos(X)
self.register_buffer('P', P)
```

forward 逻辑：

```python
x = x + self.P[:, :x.shape[1], :]
return self.dropout(x)
```

测试配置：

```python
encoding_dim = 32
num_steps = 60
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
```

可视化内容：

```python
plot(
    torch.arange(num_steps),
    P[0, :, 6:10],
    xlabel='Row (position)',
    legend=["Col %d" % d for d in range(6, 10)]
)
```

---

## 今日实验

### 实验 1：全 0 输入验证位置编码

使用：

```python
X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
```

因为输入为全 0，所以输出主要体现 positional encoding 本身。

结论：

> 当 token embedding 为 0 时，输出可以直接观察位置编码的数值结构。

### 实验 2：可视化不同维度的位置编码波形

可视化：

```python
P[0, :, 6:10]
```

观察不同列在 position 维度上的变化。

结论：

> 不同维度对应不同频率的 sin / cos 波形，从而为不同位置提供可区分的编码模式。

### 实验 3：shape 验证

输入：

```text
[1, 60, 32]
```

位置编码：

```text
[1, 60, 32]
```

输出：

```text
[1, 60, 32]
```

结论：

> 位置编码不会改变输入输出 shape。

---

## 今日考校

### Q1：为什么 Self-Attention 本身不知道 token 顺序？

参考答案：

> Self-Attention 主要通过 `QK^T` 计算 token 之间的相关性，这个过程没有显式使用 token 的位置编号。因此，如果不额外加入位置编码，模型很难区分相同 token 在不同排列顺序下的不同语义。

### Q2：为什么 position encoding 可以和 token embedding 直接相加？

参考答案：

> 因为二者具有相同的 embedding 维度，并且 position encoding 会被截取为 `[1, seq_len, embed_dim]`，可以通过 broadcasting 与 `[batch_size, seq_len, embed_dim]` 的 token embedding 相加。相加后 shape 仍然是 `[batch_size, seq_len, embed_dim]`。

### Q3：为什么位置编码加入后输出 shape 不变？

参考答案：

> 位置编码不是拼接到 embedding 后面，而是与 token embedding 做逐元素相加。因此它注入了位置信息，但不会改变 embedding 维度，也不会改变 batch size 或 seq_len。

### Q4：为什么 ViT 也需要 position embedding？

参考答案：

> ViT 会把图像切成 patch tokens。Self-Attention 可以建模 patch 与 patch 之间的关系，但如果没有位置编码，模型不知道每个 patch 在图像中的空间位置。因此 ViT 也需要 position embedding 来注入空间位置信息。

---

## 遇到的问题

### 1. sin / cos 公式不需要死记

今天只需要理解：不同 position 和不同维度会得到不同频率的编码模式。公式细节后续可以在需要时再复习。

### 2. 当前实现是固定位置编码

当前实现的是 sinusoidal positional encoding，不需要训练参数。后续在学习 ViT 或 BERT 时，还需要理解 learnable position embedding。

### 3. 可视化依赖 matplotlib

当前代码使用 `matplotlib.pyplot` 可视化位置编码波形。如果在无图形界面的服务器中运行，可能需要改为保存图片而不是 `plt.show()`。

### 4. 暂不学习 RoPE / ALiBi

RoPE、ALiBi 等相对位置编码方法暂时不作为本阶段重点。当前阶段先掌握绝对位置编码和 ViT patch token 的位置需求。

---

## 明日计划

明天进入 ViT Patch Embedding。

### 理论目标

重点理解：

1. 图像如何被切成 patch tokens；
2. 为什么 ViT 可以把图像当作序列输入 Transformer；
3. patch size 如何影响 token 数量；
4. patch embedding 和文本 token embedding 的类比关系；
5. class token 和 position embedding 的作用。

### 代码目标

创建：

```text
models/patch_embedding.py
```

最低实现：

- 输入图像 shape：`[batch_size, channels, height, width]`；
- 使用 `nn.Conv2d` 实现 patch embedding；
- 输出 patch tokens：`[batch_size, num_patches, embed_dim]`；
- 打印每一步 shape；
- 验证 `num_patches = (image_size / patch_size)^2`。

### 笔记目标

创建：

```text
notes/vit/patch_embedding.md
```

需要说明：

- patch embedding 的作用；
- Conv2d 实现 patchify 的原理；
- patch token 和文本 token 的对应关系；
- 为什么 ViT 需要 position embedding；
- patch size 对 token 数量和计算量的影响。

---

## 今日总结

今天完成了 Positional Encoding 的理论理解、代码实现、波形可视化和笔记沉淀。

当前 Week01 的学习链路已经推进为：

```text
Scaled Dot-Product Attention
→ Multi-Head Attention
→ Transformer Encoder Block
→ Positional Encoding
```

下一步将进入 ViT Patch Embedding，开始把 Transformer 从文本序列正式迁移到图像 token 序列。这一步是后续理解 CLIP Vision Encoder、BLIP、LLaVA 和 Qwen-VL 的关键基础。
