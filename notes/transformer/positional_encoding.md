# Positional Encoding

## 1. 为什么 Transformer 需要位置编码？

Self-Attention 本身主要计算 token 与 token 之间的相关性：

```text
QK^T → softmax → attention_weights @ V
```

这个计算过程不显式包含 token 的顺序信息。如果只给模型 token embedding，而不加入位置信息，那么模型很难区分相同 token 在不同顺序下的语义差异。

例如：

```text
我 爱 你
你 爱 我
```

这两个句子包含相同 token，但语义不同，差异来自 token 的顺序。

因此，Transformer 需要显式给 token embedding 注入位置信息。

---

## 2. Positional Encoding 的作用

Positional Encoding 的核心作用是：

> 让模型同时知道 token 是什么，以及 token 在哪里。

输入给 Transformer 的表示通常可以写成：

```text
input_embedding = token_embedding + position_embedding
```

其中：

- token embedding 表示 token 的语义内容；
- position embedding / positional encoding 表示 token 的位置。

---

## 3. Shape 理解

假设输入 token embedding 为：

```text
[batch_size, seq_len, embed_dim]
```

位置编码通常构造为：

```text
[1, max_len, embed_dim]
```

在 forward 过程中根据当前序列长度截取：

```python
self.P[:, :x.shape[1], :]
```

得到：

```text
[1, seq_len, embed_dim]
```

然后与输入相加：

```text
[batch_size, seq_len, embed_dim] + [1, seq_len, embed_dim]
```

通过 broadcasting 得到：

```text
[batch_size, seq_len, embed_dim]
```

因此位置编码只注入位置信息，不改变输入输出 shape。

---

## 4. Sinusoidal Positional Encoding

Sinusoidal Positional Encoding 使用固定的 sin / cos 函数生成位置向量，不需要训练参数。

基本形式是：

```text
偶数维：sin(position / 10000^(2i / d_model))
奇数维：cos(position / 10000^(2i / d_model))
```

在代码中对应：

```python
P[:, :, 0::2] = torch.sin(X)
P[:, :, 1::2] = torch.cos(X)
```

其中：

- `0::2` 表示偶数维；
- `1::2` 表示奇数维；
- 不同维度对应不同频率的正弦 / 余弦波。

---

## 5. register_buffer 的作用

位置编码矩阵 `P` 不是需要训练的参数，但它需要随着模型一起保存，并且在模型迁移到 GPU 时自动迁移到同一设备。

因此使用：

```python
self.register_buffer('P', P)
```

这表示：

- `P` 不会作为可训练参数更新；
- `P` 会进入 `state_dict`；
- `model.to(device)` 时，`P` 会一起移动到对应设备。

---

## 6. Dropout 的作用

在位置编码后加入 dropout：

```python
return self.dropout(x)
```

可以对输入表示进行一定正则化，降低模型对某些固定位置模式的过度依赖。

---

## 7. 和 ViT 的关系

ViT 会把图像切分成 patch tokens。

如果没有位置编码，模型只知道图像中有哪些 patch，但不知道这些 patch 在图像中的空间位置。

例如：

- 左上角 patch；
- 右下角 patch；
- 中心区域 patch。

这些 patch 的空间关系对图像理解非常重要，因此 ViT 也需要 position embedding。

---

## 8. 今日总结

Positional Encoding 的核心作用是：

> 给 token embedding 注入顺序或空间位置信息。

它保证 Transformer 不仅能建模 token 之间的关系，还能理解 token 的排列顺序或空间位置。
