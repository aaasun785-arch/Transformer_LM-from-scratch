from train_BPE import train_bpe
INPUT_PATH="data/TinyStoriesV2-GPT4-train.txt"
def print_outputs(vocab:dict[int,bytes],
                  
                  ):
    longest_token=None
    longest_length=-1
    longest_id=-1
    for index,tok in vocab.items():
        length=len(tok)
        if length>longest_length:
            longest_length=length
            longest_token=tok
            longest_id=index
    print(f"longest token_id:{longest_id}")
    print(f"longest token:{longest_token.decode("utf-8")}")
    print(f"longest length:{longest_length}")

if "__name__"=="__main__":
    vocab,merge=train_bpe(INPUT_PATH,10000,special_tokens = ["<|endoftext|>"])
    print_outputs(vocab)