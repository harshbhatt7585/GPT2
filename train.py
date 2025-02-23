import torch
import torch.nn as nn
from transformer import GPT2
from torch.nn import functional as F
from dataset import prepare_gpt2_dataset

class GPTConfig:
    def __init__(self):
        self.n_layer = 8
        self.d_embed = 768
        self.n_heads = 4
        self.vocab_size = 5025
        self.n_positions = 1024
        self.n_ctx = 512
        self.layer_norm_epsilon = 1e-5

lr = 1e-5
EPOCHS = 10

config = GPTConfig()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = GPT2(config).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr)

total_params = sum(p.numel() for p in model.parameters())

dataloader, tokenizer = prepare_gpt2_dataset(batch_size=2)

past = None
for epoch in range(EPOCHS):
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(device)


        input_ids = batch[:, :-1] 
        targets = batch[:, 1:]  
        if input_ids.shape[1] == 0:
            continue
    
        logits, presents = model(input_ids)  

        loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))  
        
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")
