import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from attention import attention

class Multi_head_attention(nn.Module):
    """
    输入：[batch_size,seq_len,embed_dim]
        一个批次中有 batch_size 个句子，每个句子有 seq_len 个单词，每个单词由 embed_dim 维向量表示
    输出：[batch_size,seq_len,embed_dim]
        经过多头注意力处理后的特征表示，维度与输入保持一致，方便残差连接
    """
    def __init__(self,embed_dim,nums_head):
        super(Multi_head_attention, self).__init__()
        assert embed_dim % nums_head == 0, "embed_dim must be divisible by nums_head"
        self.embed_dim=embed_dim
        self.nums_head=nums_head
        self.head_dim=embed_dim//nums_head

        #定义线性变换层
        self.W_q=nn.Linear(embed_dim,embed_dim)
        self.W_k=nn.Linear(embed_dim,embed_dim)
        self.W_v=nn.Linear(embed_dim,embed_dim)
        self.W_o=nn.Linear(embed_dim,embed_dim)

    def forward(self,q,k,v):
        batch_size=q.size(0)

        #1.线性变换并拆分为多头
        Q=self.W_q(q).view(batch_size,-1,self.nums_head,self.head_dim).transpose(1,2)
        K=self.W_k(k).view(batch_size,-1,self.nums_head,self.head_dim).transpose(1,2)
        V=self.W_v(v).view(batch_size,-1,self.nums_head,self.head_dim).transpose(1,2)

        #2.计算注意力 (调用 attention.py 中的函数)
        # x shape: [batch_size, nums_head, seq_len, head_dim]
        x, attn = attention(Q, K, V)

        #3.合并多头 (Concatenate)
        # transpose 换回: [batch_size, seq_len, nums_head, head_dim]
        # view 合并: [batch_size, seq_len, embed_dim]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        #4.最后的线性变换
        output = self.W_o(x)
        
        return output, attn
    
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(2, 6, 64)  # B=2, L=6, D=64 (4 heads × 16 dim)
    mha = Multi_head_attention(embed_dim=64, nums_head=4)
    out, weights = mha(x, x, x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Attn weights shape: {weights.shape}")