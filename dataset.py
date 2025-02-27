import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from encoder import Encoder, get_encoder

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        
        for text in texts:
            input_ids = tokenizer.encode(text)
            self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]
    
def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.
    """
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0) 

def prepare_gpt2_dataset(dataset_name="wikitext", subset="wikitext-2-raw-v1", batch_size=8):
    """
    Fetches and preprocesses a dataset from Hugging Face.
    - dataset_name: Name of the dataset (e.g., "wikitext", "openwebtext").
    - subset: Specific subset of the dataset (if applicable).
    - batch_size: Batch size for training.
    
    Returns a PyTorch DataLoader.
    """

    dataset = load_dataset(dataset_name, subset)
    
    tokenizer = get_encoder()

    texts = dataset["train"]["text"]

    tokenized_dataset = GPT2Dataset(texts, tokenizer)
    
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    return dataloader, tokenizer

# Example usage
if __name__ == "__main__":
    dataloader, tokenizer = prepare_gpt2_dataset()

    
    # Check a sample batch
    for idx, batch in enumerate(dataloader):
        print(batch.shape)  # Expected: (batch_size, sequence_length)
        if idx >= 3:
            break