import torch 
import torch.nn as nn
import math
import matplotlib.pyplot as plt

def plot(x, y, xlabel=None, ylabel=None, legend=None, figsize=(6, 2.5)):
    """简单的绘图函数，用于可视化位置编码波形"""
    plt.figure(figsize=figsize)
    plt.plot(x.detach().numpy(), y.detach().numpy())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)
    plt.show()

class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len=1000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        # 创建位置编码矩阵 P
        P=torch.zeros((1,max_len,num_hiddens))
        X=torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        P[:,:,0::2]=torch.sin(X)
        P[:,:,1::2]=torch.cos(X)
        # 使用 register_buffer，这样 P 会自动随模型移动到对应设备（CPU/GPU）
        self.register_buffer('P', P)

    def forward(self,x):
        # 修正变量名，从 P 中截取与输入 x 相同长度的序列
        x=x+self.P[:,:x.shape[1],:]
        return self.dropout(x)
    
if __name__=="__main__":
    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    
    # 生成全 0 输入，经过编码后只剩下位置信息 P
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    
    # 可视化第 6, 7, 8, 9 列 (不同频率的正弦和余弦波形)
    plot(torch.arange(num_steps), P[0, :, 6:10], 
         xlabel='Row (position)', legend=["Col %d" % d for d in range(6, 10)])