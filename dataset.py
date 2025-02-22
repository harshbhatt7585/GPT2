import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        
        for text in texts:
            tokens = tokenizer(
                text, 
                max_length=self.max_length, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            self.input_ids.append(tokens["input_ids"].squeeze(0))  # Remove batch dimension

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]

def prepare_gpt2_dataset(dataset_name="wikitext", subset="wikitext-2-raw-v1", batch_size=8):
    """
    Fetches and preprocesses a dataset from Hugging Face.
    - dataset_name: Name of the dataset (e.g., "wikitext", "openwebtext").
    - subset: Specific subset of the dataset (if applicable).
    - batch_size: Batch size for training.
    
    Returns a PyTorch DataLoader.
    """

    dataset = load_dataset(dataset_name, subset)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    

    texts = dataset["train"]["text"]

    tokenized_dataset = GPT2Dataset(texts, tokenizer)
    
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, tokenizer

# Example usage
if __name__ == "__main__":
    dataloader, tokenizer = prepare_gpt2_dataset()
    
    # Check a sample batch
    for batch in dataloader:
        print(batch.shape)  # Expected: (batch_size, sequence_length)
        break