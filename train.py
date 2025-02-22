import torch
import torch.nn as nn 
from attention import MultiHeadAttention
from transformer import GPT2
    
class GPTConfig:
    def __init__(self):
        self.n_layer = 12           # Number of transformer blocks
        self.d_embed = 768          # Embedding dimension
        self.n_heads = 12           # Number of attention heads
        self.vocab_size = 50257     # Vocabulary size (same as GPT-2)
        self.n_positions = 1024     # Maximum sequence length
        self.n_ctx = 1024           # Context size (same as n_positions)
        self.layer_norm_epsilon = 1e-5  # LayerNorm epsilon


def train_gpt2():
    config = GPTConfig()
