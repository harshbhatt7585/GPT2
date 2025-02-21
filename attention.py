import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F  # Fixed incorrect import

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_embed):
        super().__init__()
        self.n_heads = n_heads
        self.d_heads = d_embed // n_heads  # Head size
        self.in_proj = nn.Linear(d_embed, 3 * d_embed)
        self.out_proj = nn.Linear(d_embed, d_embed)
    
    def forward(self, x, layer_past=None):
        # x: (batch, sequence, features)
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)

        x = self.in_proj(x)  # (b, s, 3*f)
        
        # 3 * (b, s, f)
        q, k, v = x.chunk(3, dim=-1)
        
        # (b, h, s, f / h)
        q = q.view(interim_shape).transpose(1, 2)  # (b, h, s, d_h)
        k = k.view(interim_shape).transpose(1, 2)  # (b, h, s, d_h)
        v = v.view(interim_shape).transpose(1, 2)  # (b, h, s, d_h)

        # === Layer Past Caching ===
        if layer_past is not None:
            past_key, past_value = layer_past  # Unpack past k, v
            k = torch.cat((past_key, k), dim=-2)  # Append along sequence dim
            v = torch.cat((past_value, v), dim=-2)  # Append along sequence dim
        
        present = (k, v)  # Store for next time step

        # Scaled Dot-Product Attention
        weight = q @ k.transpose(-1, -2)  # (b, h, s, s)
        weight /= math.sqrt(self.d_heads)

        # Causal Mask for autoregressive decoding
        mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
        weight.masked_fill_(mask, -torch.inf)

        weight = F.softmax(weight, dim=-1)
        output = weight @ v  # (b, h, s, d_h)

        # Reshape back
        output = output.transpose(1, 2).reshape(input_shape)  # (b, s, f)
        output = self.out_proj(output)  # Final projection

        return output, present  # Return updated past k, v