import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block 实现
    逻辑顺序：Multi-Head Attention -> Add & Norm -> Feed Forward Network -> Add & Norm
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 1. 初始化多头注意力机制
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)

        # 2. 初始化两个 LayerNorm 层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 增加 Dropout 层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 3. 初始化 FFN (Sequential)
        # 提示：ff_dim 建议设为 embed_dim 的 4 倍 (如 64 -> 256)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        # --- Post-LN 逻辑 (原始 Transformer 论文方案) ---

        # 1. 计算 Self-Attention
        attn_out, attn_weights = self.mha(x, x, x)

        # 2. 残差连接 + LayerNorm
        # 公式：x = LayerNorm(x + Sublayer(x))
        x = self.norm1(x + attn_out)
        x = self.norm1(x + self.dropout1(attn_out))

        # 3. 经过 Feed Forward Network
        ffn_out = self.ffn(x)

        # 4. 残差连接 + LayerNorm
        x = self.norm2(x + ffn_out)
        x = self.norm2(x + self.dropout2(ffn_out))

        # 返回: x, attn_weights
        return x, attn_weights

if __name__ == "__main__":
    # 测试代码
    torch.manual_seed(42)
    sample_input = torch.randn(2, 6, 64) # [B, L, D]
    encoder_block = TransformerEncoderBlock(embed_dim=64, num_heads=8, ff_dim=256)
    
    out, weights = encoder_block(sample_input)
    print(f"\nFinal Output Shape: {out.shape}") # 应该保持 [2, 6, 64]
    print(f"Attention Weights Shape: {weights.shape}") # [2, 8, 6, 6]
    print(weights.sum(dim=-1))