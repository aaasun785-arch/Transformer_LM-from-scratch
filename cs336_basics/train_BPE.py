import os
import regex as re 
from collections import defaultdict
from typing import Dict,Tuple,Union,BinaryIO,Set
import multiprocessing as mp

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk_for_word_counts(file_path:str,
                                  start:int,
                                  end:int,
                                  special_tokens:list[str]):
    with open(file_path,"rb") as f:
        f.seek(start)
        data=f.read(end-start)
        text=data.decode("utf-8")
    if special_tokens:
        escaped='|'.join(re.escape(tok) for tok in special_tokens)
        split_re=re.compile(f'({escaped})')
        segments=split_re.split(text)
    else:
        segments=[text]
    word_cnt=defaultdict(int)
    for seg in segments:
        if not seg or seg in special_tokens:
            continue
        for m in re.finditer(PAT,seg):
            word_bytes=m.group().encode("utf-8")
            word_tuple=tuple(bytes([b]) for b in word_bytes)
            if len(word_tuple)>=2:
                word_cnt[word_tuple]+=1
    return word_cnt

def merge_word(word:tuple[bytes,...],
               pair:tuple[bytes,bytes]):
    merged=pair[0]+pair[1]
    i=0
    out=[]
    while i<len(word):
        if(i+1<len(word) and word[i]==pair[0] and word[i+1]==pair[1]):
            out.append(merged)
            i+=2
        else:
            out.append(word[i])
            i+=1
    return out

def train_bpe(input_path:Union[str,os.PathLike],
              vocab_size:int,
              special_tokens:list[str],
              num_workers:4):
    vocab={i:bytes([i]) for i in range(256)}
    next_id=256
    special_ids={}
    for tok in special_tokens:
        tok_bytes=tok.encode("utf-8")
        vocab[next_id]=tok_bytes
        special_ids[tok_bytes]=next_id
        next_id+=1

    special_ids_set=set(special_ids.keys())
    base_vocab_size=len(vocab)
    #需要merge的总次数
    n_merges=max(0,vocab_size-base_vocab_size)
    #多线程处理
    split_token=special_tokens[0].encode("utf-8") if special_tokens else b''
    with open(input_path,"rb") as f:
        if split_token:
            boundaries=find_chunk_boundaries(f,num_workers,split_token)
        else:
            f.seek(0,os.SEEK_END)
            file_size=f.tell()
            f.seek(0)
            chunk_size=file_size//num_workers
            boundaries=[i*chunk_size for i in range(num_workers+1)]
            boundaries[-1]=file_size

    """ with mp.Pool(processes=num_workers)as pool: """
    args=[
            (input_path,start,end,special_tokens)
            for start,end in zip(boundaries[:-1],boundaries[1:])#zip()表示配对一个区间

        ]
    results=[process_chunk_for_word_counts(*arg)for arg in args]
    #从多线程中统计总词频
    word_cnt=defaultdict(int)
    for result in results:
        for word,count in result.items():
            word_cnt[word]+=count

    words=list(word_cnt.keys())
    wcnt=[word_cnt[w] for w in words]

    pair_freq=defaultdict(int)
    pair_occ=defaultdict(set)

    for wid,(word,count) in enumerate(word_cnt.items()):
        if len(word)<2:
            continue
        word_pairs=[(word[i],word[i+1]) for i in range(len(word)-1)]
        for p in word_pairs:
            #只统计非特殊token的匹配情况
            if p[0] not in special_ids_set and p[1] not in special_ids_set:
                pair_freq[p]+=count
        for p in word_pairs:
            if p[0] not in special_ids_set and p[1] not in special_ids_set:
                pair_occ[p].add(wid)
    #merge
    merges=[]
    for step in range(n_merges):
        best_pair=None
        best_count=-1

        for pair,count in pair_freq.items():
            if count<0:
                continue
            elif count>best_count:
                best_count=count
                best_pair=pair
            elif count==best_count:
                if pair[0]>best_pair[0] or (pair[0]==best_pair[0] and pair[1]>best_pair[1]):
                    best_pair=pair

        if best_pair==None or best_count==-1:
            break

        merges.append(best_pair)

        new_token_id=base_vocab_size+step
        new_bytes=best_pair[0]+best_pair[1]
        vocab[new_token_id]=new_bytes

        affected=list(pair_occ.get(best_pair,[]))
        if not affected:
            continue
        for wid in affected:
            old_word=words[wid]
            if len(old_word)<2:
                continue
            count=wcnt[wid]

            for i in range(len(old_word)-1):
                p=(old_word[i],old_word[i+1])
                if p[0] in special_ids_set or p[1] in special_ids_set:
                    continue
                pair_freq[p]-=count
                occ=pair_occ.get(p)
                if occ is not None:
                    occ.discard(wid)
            #进行合并
            new_word=merge_word(old_word,best_pair)
            words[wid]=new_word
            #添加新pair
            for i in range(len(new_word)-1):
                p=(new_word[i],new_word[i+1])
                if p[0] in special_ids_set or p[1] in special_ids_set:
                    continue
                pair_freq[p]=pair_freq.get(p,0)+count
                pair_occ.setdefault(p,set()).add(wid)

    return vocab,merges