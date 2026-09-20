import os
import numpy as np

from cs336_basics.bpe_tokenizer import Tokenizer


VOCAB_PATH = "data/tinystories_vocab.pkl"
MERGES_PATH = "data/tinystories_merges.pkl"

TRAIN_TEXT = "data/TinyStoriesV2-GPT4-train.txt"
VAL_TEXT = "data/TinyStoriesV2-GPT4-valid.txt"

TRAIN_BIN = "data/tinystories_train.bin"
VAL_BIN = "data/tinystories_valid.bin"

SPECIAL_TOKENS = ["<|endoftext|>"]

FLUSH_SIZE = 1_000_000


def encode_file(
    input_path: str,
    output_path: str,
    tokenizer: Tokenizer,
):
    buffer = []
    total_tokens = 0

    print(f"Encoding {input_path}")
    print(f"Output -> {output_path}")

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f_in, open(output_path, "wb") as f_out:

        for token_id in tokenizer.encode_iterable(f_in):
            buffer.append(token_id)

            if len(buffer) >= FLUSH_SIZE:
                arr = np.asarray(
                    buffer,
                    dtype=np.uint16,
                )

                arr.tofile(f_out)

                total_tokens += len(buffer)
                buffer.clear()

                print(
                    f"{total_tokens:,} tokens written"
                )

        if buffer:
            arr = np.asarray(
                buffer,
                dtype=np.uint16,
            )

            arr.tofile(f_out)

            total_tokens += len(buffer)

    print(
        f"Finished: {total_tokens:,} tokens"
    )

    size_gb = os.path.getsize(output_path) / 1024**3

    print(
        f"File size: {size_gb:.2f} GB"
    )


def main():

    tokenizer = Tokenizer.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        special_tokens=SPECIAL_TOKENS,
    )

    print(
        f"Tokenizer vocab size: "
        f"{len(tokenizer.vocab)}"
    )

    encode_file(
        TRAIN_TEXT,
        TRAIN_BIN,
        tokenizer,
    )

    encode_file(
        VAL_TEXT,
        VAL_BIN,
        tokenizer,
    )


if __name__ == "__main__":
    main()