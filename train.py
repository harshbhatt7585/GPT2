import torch
import torch.nn as nn
from torch.nn import functional as F

from transformer import GPT2
from dataset import prepare_gpt2_dataset, prepare_validation_dataset
from encoder import Encoder, get_encoder


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
        self.batch_size = 4
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.wandb = True  # Set to True to enable Weights & Biases


config = GPTConfig()

if config.wandb:
    import wandb

    wandb.init(
        project="gpt2-training",
        name="experiment-3",
        config={k: v for k, v in vars(config).items() if not callable(v) and not k.startswith("__")},
    )

model = GPT2(config).to(config.device)
criterion = nn.CrossEntropyLoss().to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
encoder = get_encoder()

# Prepare datasets
dataloader, tokenizer = prepare_gpt2_dataset(batch_size=config.batch_size)
val_dataloader = prepare_validation_dataset(batch_size=config.batch_size)

def evaluate(model, val_dataloader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for val_batch in val_dataloader:
            val_batch = val_batch.to(config.device)
            input_ids = val_batch[:, :-1]
            targets = val_batch[:, 1:]
            if input_ids.shape[1] == 0:
                continue

            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    return avg_val_loss


for epoch in range(config.epochs):
    model.train()
    epoch_loss = 0
    average_batch_loss = 0

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
        average_batch_loss += loss.item()

        if config.wandb and idx % 50 == 0:
            wandb.log(
                {
                    "batch_loss": average_batch_loss / 50,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            print(f"Epoch [{epoch+1}/{config.epochs}], Batch [{idx+1}/{len(dataloader)}], Average Loss over 50 batch: {average_batch_loss / 50}")
            average_batch_loss = 0

    # Evaluate on validation set
    val_loss = evaluate(model, val_dataloader)
    print(f"Epoch [{epoch+1}/{config.epochs}], Validation Loss: {val_loss}")

    if config.wandb:
        wandb.log({
            "epoch_loss": epoch_loss / len(dataloader),
            "validation_loss": val_loss
        })
    
    torch.save(model.state_dict(), f'models/gpt2_check_{epoch}.pt')

if config.wandb:
    wandb.finish()