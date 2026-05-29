import math

import torch


def scaled_dot_product_attention(query, key, value):
    """最小版 scaled dot-product attention。"""
    # query: [batch_size, seq_len_q, d_k]
    # key: [batch_size, seq_len_k, d_k]
    # value: [batch_size, seq_len_k, d_v]
    d_k = query.shape[-1]

    # 计算注意力分数：[batch_size, seq_len_q, seq_len_k]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 对最后一维做 softmax，得到每个 token 对其他 token 的关注权重
    attention_weights = torch.softmax(scores, dim=-1)

    # 使用注意力权重对 value 加权求和
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4
    d_k = 8
    d_v = 8

    # 构造一组最小示例输入
    query = torch.randn(batch_size, seq_len, d_k)
    key = torch.randn(batch_size, seq_len, d_k)
    value = torch.randn(batch_size, seq_len, d_v)

    output, attention_weights = scaled_dot_product_attention(query, key, value)

    print("query shape:", query.shape)
    print("key shape:", key.shape)
    print("value shape:", value.shape)
    print("attention_weights shape:", attention_weights.shape)
    print("output shape:", output.shape)
