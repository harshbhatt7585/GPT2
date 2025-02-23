from dataset import load_dataset
from encoder import Encoder



def prepare_encoder_dataset(dataset_name="wikitext", subset="wikitext-2-raw-v1"):
    dataset = load_dataset(dataset_name, subset)
    
    tokenizer = Encoder()

    texts = dataset["train"]["text"][:100]
    text_corpus = "".join([text for text in texts])

    with open("corpus.txt", 'w') as f:
        f.write(text_corpus)

    tokenizer.train(['corpus.txt'])


if __name__ == "__main__":
    prepare_encoder_dataset()