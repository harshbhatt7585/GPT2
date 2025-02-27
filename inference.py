import torch
from transformer import GPT2
from GPT2.encoder import Encoder
import torch.nn.functional as F
from tqdm import trange  # Progress bar (optional)

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

# Load Model & Tokenizer
model = GPT2(config)
encoder = Encoder()

# Load Pretrained Weights
model.load_state_dict(torch.load('/Users/harshbhatt/Projects/implementations/GPT2/models/gpt2_check_0.pt', map_location=config.device))
model = model.to(config.device)
model.eval()


length = 50  
temperature = 1.0
top_k = 0
sample = True
device = config.device


prompt = "The game began development in 2010"
context = torch.tensor(encoder.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)


prev = context
output = context
past = None

# Start Generating Tokens
with torch.no_grad():
    for i in trange(length):
        logits, past = model(prev, past=past) 
        logits = logits[:, -1, :] / temperature 

        if top_k > 0:
            values, indices = torch.topk(logits, k=top_k, dim=-1)
            min_value = values[:, -1]
            logits[logits < min_value] = float('-inf')

        log_probs = F.softmax(logits, dim=-1)

        if sample:
            prev = torch.multinomial(log_probs, num_samples=1)
        else:
            _, prev = torch.topk(log_probs, k=1, dim=-1)

  
        output = torch.cat((output, prev), dim=1)


generated_text = encoder.decode(output.squeeze(0).tolist())
print("\nGenerated Text:\n", generated_text)