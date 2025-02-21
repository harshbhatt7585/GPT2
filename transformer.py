import torch.nn as nn
import copy 
from attention import MultiHeadAttention
import torch

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
        self.layer_norm_1 = nn.LayerNorm(d_emebd, eps=config.layer_norm_epsilon)
        self.attention = MultiHeadAttention(config.n_heads, d_emebd)
        self.layer_norm_2 = nn.LayerNorm(d_emebd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * d_emebd, config)

    def forward(self, x, layer_past=None):
        x = self.layer_norm_1(x)
        residue = self.attention(x, layer_past=layer_past)
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
        self.layer_norm = nn.LayerNorm(config.d_embed, eps=config.layer_norm_epsilon)

    
    def set_mbeddings_weights(self, model_embedding_weights):
        embed_shape = model_embedding_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=True)
        self.decoder.weight = model_embedding_weights


    def forward(self, input_ids, position_ids=None, token_type_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.blocks)
        else:
            past_length = past[0][0].size(-2)
        
        if position_ids is None:
            position_ids = torch.arrange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        
        input_ids = input_ids.view(-1, input_shape.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        text_embeddings = self.text_embeddding(input_ids)
        positional_embeddings = self.positional_embedding(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeddings = self.text_embeddding(token_type_ids)
        else:
            token_type_embeddings = 0

        hidden_states = text_embeddings + positional_embeddings + token_type_embeddings
        presents = []
        for block, layer_past in zip(self.blocks, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        hidden_states = self.layer_norm(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents
    


class GPT2Head(nn.Module):
    def __init__(self, model_embedding_weights, config):
        super(GPT2Head, self).__init__()
        self.d_embed = config.d_embed
        self.set_embedding_weights(model_embedding_weights)
    
    def set_mbeddings_weights(self, model_embedding_weights):
        embed_shape = model_embedding_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=True)
        self.decoder.weight = model_embedding_weights
    
    def forward(self, hidden_state):
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2(config)
        self.lm_head = GPT2Head(self.transformer.text_embeddding.weight, config)

    def set_teid(self):
        self.lm_head.set_mbeddings_weights(self.transformer.text_embeddding.weight)
    
    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, position_ids, token_type_ids, past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=1)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents