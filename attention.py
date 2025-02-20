import math 
import torch 
import torch.nn as nn
from torch.nn import function as F


class Attention(nn.Module):
    def __init__(self, n_heads, d_embed):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Conv1d(d_embed, 3 * d_embed, kernel_size=1, stride=1, padding=0)
        self.out_proj = nn.Conv1d(d_embed, d_embed, kernel_size=1, stride=1, padding=0)
        
        self.d_heads = d_embed // n_heads
    
    
    def forward(self, x):
        # x: (batch, sequence, features)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        interim_shape = (batch_size, seq_len, self.n_heads, d_embed)

        x = self.in_proj(x) # (b, s, f * 3)
        
        # 3 * (b, s, f)
        q, k, v = x.chunk(3, dim=-1)
        
        # (b, h, s, f / h)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_heads)

        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        weight.masked_fill_(mask, -torch.inf)

        weight = F.softmax(weight, dim=-1)
        output = weight @ v

        output = output.transpose(1, 2)
        output = output.reshape(input_shape)

        output = self.out_proj(output)
        return output


        
        
        
        
        
        