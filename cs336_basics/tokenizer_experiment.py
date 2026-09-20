import os
import json
import time
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np

from bpe_tokenizer import Tokenizer  # 改成你的 Tokenizer 所在模块

# ============ 配置：按实际路径修改 ============
TINYSTORIES_TRAIN = "data/owt_train.txt"
TINYSTORIES_VALID = "data/owt_valid.txt"
OPENWEBTEXT_TRAIN = "data/TinyStoriesV2-GPT4-train.txt"
OPENWEBTEXT_VALID = "data/TinyStoriesV2-GPT4-valid.txt"

TINYSTORIES_VOCAB = "tokenizers/tinystories_vocab.pkl"
TINYSTORIES_MERGES = "tokenizers/tinystories_merges.pkl"
OPENWEBTEXT_VOCAB = "tokenizers/openwebtext_vocab.pkl"
OPENWEBTEXT_MERGES = "tokenizers/openwebtext_merges.pkl"

SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_DIR = "encoded"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============ 通用工具 ============
def read_documents(file_path: str, max_docs: int | None = None) -> Iterator[str]:
    """
    逐文档读取。假设：
    - TinyStories：每行一个纯文本故事
    - OpenWebText：每行一个 JSON，字段为 "text"
    如果你的格式不同，改这里即可。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content") or ""
                except json.JSONDecodeError:
                    text = line
            else:
                text = line

            if text:
                yield text
                count += 1
                if max_docs is not None and count >= max_docs:
                    break


def sample_docs(file_path: str, n: int = 10) -> List[str]:
    return list(read_documents(file_path, max_docs=n))


def compression_ratio(tokenizer: Tokenizer, docs: List[str]):
    """返回 (bytes/token, total_bytes, total_tokens)"""
    total_bytes = 0
    total_tokens = 0
    for text in docs:
        total_bytes += len(text.encode("utf-8"))
        ids = tokenizer.encode(text)
        total_tokens += len(ids)
    ratio = total_bytes / total_tokens if total_tokens else 0.0
    return ratio, total_bytes, total_tokens


def throughput(tokenizer: Tokenizer, texts: List[str]):
    """返回 (bytes/s, total_bytes, elapsed_seconds)"""
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)
    start = time.perf_counter()
    for _ in tokenizer.encode_iterable(texts):
        pass
    elapsed = time.perf_counter() - start
    rate = total_bytes / elapsed if elapsed > 0 else float("inf")
    return rate, total_bytes, elapsed


def encode_and_save_npy(tokenizer: Tokenizer, input_path: str, output_path: str, add_eot: bool = True):
    """
    内存友好版：边编码边写入二进制文件，最后可用 np.fromfile 读取。
    如果你确定数据不大，也可以先收集到 list 再 np.save。
    """
    eot_id = None
    if add_eot and "<|endoftext|>" in tokenizer.special_token_ids:
        eot_id = tokenizer.special_token_ids["<|endoftext|>"]

    with open(output_path, "wb") as f:
        for text in read_documents(input_path):
            ids = tokenizer.encode(text)
            if eot_id is not None:
                ids.append(eot_id)
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f)

    # 统计一下写了多少 token
    file_size = os.path.getsize(output_path)
    num_tokens = file_size // np.dtype(np.uint16).itemsize
    print(f"Saved {output_path}: {num_tokens} tokens, dtype=uint16")
    return num_tokens


# ============ 主实验 ============
if __name__ == "__main__":
    # 加载两个 tokenizer
    ts_tok = Tokenizer.from_files(
        TINYSTORIES_VOCAB, TINYSTORIES_MERGES, special_tokens=SPECIAL_TOKENS
    )
    owt_tok = Tokenizer.from_files(
        OPENWEBTEXT_VOCAB, OPENWEBTEXT_MERGES, special_tokens=SPECIAL_TOKENS
    )

    # ---------- (a) 采样 10 篇，计算压缩比 ----------
    ts_docs = sample_docs(TINYSTORIES_TRAIN, 10)
    owt_docs = sample_docs(OPENWEBTEXT_TRAIN, 10)

    ts_ratio, ts_b, ts_t = compression_ratio(ts_tok, ts_docs)
    owt_ratio, owt_b, owt_t = compression_ratio(owt_tok, owt_docs)

    print("\n=== (a) 压缩比 ===")
    print(f"TinyStories tokenizer on TinyStories: {ts_ratio:.3f} bytes/token "
          f"({ts_b} bytes, {ts_t} tokens)")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.3f} bytes/token "
          f"({owt_b} bytes, {owt_t} tokens)")

    # ---------- (b) 用 TinyStories tokenizer 编码 OpenWebText ----------
    owt_with_ts_ratio, owt_ts_b, owt_ts_t = compression_ratio(ts_tok, owt_docs)

    print("\n=== (b) 跨 tokenizer ===")
    print(f"TinyStories tokenizer on OpenWebText: {owt_with_ts_ratio:.3f} bytes/token "
          f"({owt_ts_b} bytes, {owt_ts_t} tokens)")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_ratio:.3f} bytes/token")

    # ---------- (c) 吞吐量 & Pile 时间 ----------
    # 取约 10MB 文本测吞吐
    sample_texts = []
    total_bytes = 0
    target = 10 * 1024 * 1024  # 10 MB
    for text in read_documents(OPENWEBTEXT_TRAIN):
        sample_texts.append(text)
        total_bytes += len(text.encode("utf-8"))
        if total_bytes >= target:
            break

    thr, thr_bytes, thr_time = throughput(owt_tok, sample_texts)
    pile_bytes = 825 * 1024**3  # 825 GB
    pile_seconds = pile_bytes / thr
    pile_hours = pile_seconds / 3600
    pile_days = pile_hours / 24

    print("\n=== (c) 吞吐量 ===")
    print(f"Throughput: {thr:.2f} bytes/s "
          f"({thr_bytes} bytes in {thr_time:.2f}s)")
    print(f"Pile 825GB 预计耗时: {pile_seconds:.2f} s = "
          f"{pile_hours:.2f} h = {pile_days:.2f} days")

    # ---------- (d) 编码 train/dev 并保存为 uint16 ----------
    print("\n=== (d) 编码并保存 ===")
    encode_and_save_npy(
        ts_tok, TINYSTORIES_TRAIN, os.path.join(OUTPUT_DIR, "tinystories_train.bin")
    )
    encode_and_save_npy(
        ts_tok, TINYSTORIES_VALID, os.path.join(OUTPUT_DIR, "tinystories_valid.bin")
    )
    encode_and_save_npy(
        owt_tok, OPENWEBTEXT_TRAIN, os.path.join(OUTPUT_DIR, "openwebtext_train.bin")
    )
    encode_and_save_npy(
        owt_tok, OPENWEBTEXT_VALID, os.path.join(OUTPUT_DIR, "openwebtext_valid.bin")
    )

    print("\n(d) uint16 合适，因为最大词表 32K < 65536，uint16 可表示 0~65535，"
          "比 int32/int64 省一半以上内存和磁盘。")