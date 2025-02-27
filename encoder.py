from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os

class Encoder:
    def __init__(self, vocab_size=5000, special_tokens=None, model_path="encoder.json"):
        """
        Initializes the Encoder.
        - vocab_size: Number of tokens in the vocabulary.
        - special_tokens: List of special tokens (e.g., [UNK], [CLS]).
        - model_path: File path to save/load the trained tokenizer.
        """
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

        if os.path.exists(self.model_path):
            self.tokenizer = Tokenizer.from_file(self.model_path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()

            self.tokenizer.post_processor = TemplateProcessing(
                single="$A [SEP]",
                pair="$A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[SEP]", self.special_tokens.index("[SEP]"))]
            )

    def train(self, corpus_files):
        """
        Trains the tokenizer using a corpus file.
        - corpus_files: List of file paths containing training text.
        """
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train(files=corpus_files, trainer=trainer)

        self.tokenizer.save(self.model_path)
        print(f"Tokenizer trained and saved at {self.model_path}")

    def encode(self, text):
        """
        Encodes text into token IDs.
        """
        encoded = self.tokenizer.encode(text)
        return encoded.ids

    def decode(self, token_ids):
        """
        Decodes token IDs back into text.
        """
        return self.tokenizer.decode(token_ids)
    



if __name__ == "__main__":
    encoder = Encoder(vocab_size=5000)

    # corpus_file = "corpus.txt"  
    # if not os.path.exists(encoder.model_path):
    #     encoder.train([corpus_file])

    text = "Hello, how are you?"
    encoded_ids = encoder.encode(text)
    print("Encoded:", encoded_ids)

    decoded_text = encoder.decode(encoded_ids)
    print("Decoded:", decoded_text)