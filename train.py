import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer import GPT2
from dataset import prepare_gpt2_dataset
from encoder import Encoder


class GPTConfig:
    def __init__(self):
        self.n_layer = 8
        self.d_embed = 768
        self.n_heads = 4
        self.vocab_size = 5025
        self.n_positions = 1024
        self.n_ctx = 512
        self.layer_norm_epsilon = 1e-5

        self.learning_rate = 1e-5
        self.epochs = 10
        self.batch_size = 2
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.wandb = True  # Set to True to enable Weights & Biases


config = GPTConfig()

if config.wandb:
    import wandb

    wandb.init(
        project="gpt2-training",
        name="experiment-1",
        config={k: v for k, v in vars(config).items() if not callable(v) and not k.startswith("__")},
    )

model = GPT2(config).to(config.device)
criterion = nn.CrossEntropyLoss().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
encoder = Encoder()

dataloader, tokenizer = prepare_gpt2_dataset(batch_size=config.batch_size)

for epoch in range(config.epochs):
    epoch_loss = 0
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        batch = batch.to(config.device)

        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        if input_ids.shape[1] == 0:
            continue

        logits, presents = model(input_ids)

        loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if config.wandb:
            wandb.log(
                {
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(f"Epoch [{epoch+1}/{config.epochs}], Batch [{idx+1}/{len(dataloader)}], Loss: {loss.item()}")

    if config.wandb:
        wandb.log({"epoch_loss": epoch_loss / len(dataloader)})

if config.wandb:
    wandb.finish()