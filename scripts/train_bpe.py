import pickle
from cs336_basics.bpe import train_bpe
def main():
    vocab, merges = train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_workers=4,
    )
    with open("data/tinystories_vocab.pkl","wb") as f:
        pickle.dump(vocab, f)
    with open("data/tinystories_merges.pkl","wb") as f:
        pickle.dump(merges, f)
if __name__ == "__main__":
    main()