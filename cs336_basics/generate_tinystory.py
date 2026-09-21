import torch

from cs336_basics.transformer import TransformerLM,AdamW,load_checkpoint,decode
from cs336_basics.bpe_tokenizer import Tokenizer

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=8
D_FF=1344
ROPE_THETA=10000.0

CHECKPOINT_PATH="checkpoint.pt"
VOCAB_PATH="data/tinystories_vocab.pkl"
MERGES_PATH="data/tinystories_merges.pkl"


def main():
    # 1. 恢复 tokenizer
    tokenizer=Tokenizer.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        special_tokens=["<|endoftext|>"]
    )

    # 2. 创建和训练时完全相同的模型
    model=TransformerLM(
        VOCAB_SIZE,
        CONTEXT_LENGTH,
        D_MODEL,
        NUM_LAYERS,
        NUM_HEADS,
        D_FF,
        ROPE_THETA
    ).to(DEVICE)

    # 3. 加载 checkpoint
    checkpoint=torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )
    model.load_state_dict(checkpoint["model"])

    print(f"loaded checkpoint at step {checkpoint['iteration']}")

    # 4. 生成文本
    prompt="Once upon a time"

    text=decode(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        context_length=CONTEXT_LENGTH,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9
    )

    print("\n===== Generated Text =====\n")
    print(text)


if __name__=="__main__":
    main()