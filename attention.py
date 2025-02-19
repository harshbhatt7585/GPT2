import math 
import torch 
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Conv1d(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Conv1d(d_embed, d_embed, bias=out_proj_bias)
        
        self.d_heads = d_embed // n_heads

        
        
        
        
        
        