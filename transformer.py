import torch.nn as nn
import copy 
from attention import MultiHeadAttention


class MLP(nn.Module):
    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        d_embed = config.d_embed
        self.in_proj = nn.Conv1d(n_state, d_embed)
        self.out_proj = nn.Conv1d(d_embed, n_state)
        self.gelu = nn.GeLU()

    def forward(self, x):
        x = self.gelu(x)
        x = self.in_proj(x)
        x = self.out_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(TransformerBlock, self).__init__()
        d_emebd = config.d_emebd
        self.layer_norm_1 = LayerNorm(d_emebd, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadAttention(config.n_heads, d_emebd)
        self.layer_norm_2 = LayerNorm(d_emebd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * d_emebd, config)

    def forward(self, x):
        residue = self.attention(x)
        x = x + residue
        residue = self.mlp(x)
        x = x + residue
        return x


class GPT2(nn.Module):
    def __init__(self):
        super(GPT2, self).__init__()
        self.n_layer = config.n_layer
        self.d_embed = config.d_embed
        self.n_vocab = config.vocab_size

        self.text_embeddding = nn.Embedding(config.vocab_size, config.d_embed)
        self.positional_embedding = nn.Embedding(config.n_positions, config.d_embed)
        block = TransformerBlock(config.n_ctx, config, scale=True)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config.d_embed, eps=config.layer_norm_epsilon)