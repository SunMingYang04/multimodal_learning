import torch
import torch.nn.functional as F
import math

def attention(Q,K,V):
    """
    scale dot-product attention
    输入：
    Q:[batch_size, seq_len_q, d_k]
    K:[batch_size, seq_len_k，d_k]
    V:[batch_size, seq_len_k, d_v]
    输出：
    outputs:[batch_size, seq_len_q, d_v]
    attention_weights:[batch_size, seq_len_q, seq_len_k]
    """
    d_k = K.shape[-1]

    #计算dot-product
    scores=torch.matmul(Q,K.transpose(-1,-2))/math.sqrt(d_k)
    #进行softmax
    attention_weights=F.softmax(scores,dim=-1)
    #计算outputs
    outputs=torch.matmul(attention_weights,V)

    return outputs,attention_weights

if __name__ == '__main__':
    torch.manual_seed(42)
    q=k=v=torch.randn(2,4,8)

    output,attn=attention(q,k,v)
    print(f"Q: {q.shape} → Scores: {attn.shape} → Output: {output.shape}")
    print(f"\nAttention Matrix (Batch 0):\n{attn[0]}")
    print(f"Row sums (should be ~1.0): {attn[0].sum(dim=1)}")