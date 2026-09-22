import regex as re
import os
import pickle
from .bpe import merge_word
from collections import defaultdict
from typing import Iterable,Iterator

PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer():
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens if special_tokens else []
        #token2byte映射
        self.token_to_id={v:k for k,v in vocab.items()}
        #处理特殊token的id
        self.special_token_ids:dict[str,int]={}
        for token in self.special_tokens:
            token_bytes=token.encode("utf-8")
            if token_bytes in self.token_to_id:
                self.special_token_ids[token]=self.token_to_id[token_bytes]
            else:
                new_id=max(self.vocab.keys())+1 if self.vocab else 0
                self.vocab[new_id]=token_bytes
                self.token_to_id[token_bytes]=new_id
                self.special_token_ids[token]=new_id
        #用于处理merge的先后顺序
        self.merge_rank:dict[tuple[bytes,bytes],int]={}
        for i,(first,second) in enumerate(self.merges):
            self.merge_rank[(first,second)]=i

    def encode_bytes(self,text_bytes:bytes) ->list[int]:
        tokens=[bytes([b]) for b in text_bytes]
        #反复merge，应用while
        while len(tokens)>=2:
            best_rank=float("inf")
            best_index=-1
            for i in range(len(tokens)-1):
                pair=(tokens[i],tokens[i+1])
                if pair in self.merge_rank:
                    rank=self.merge_rank[pair]
                    if rank<best_rank:
                        best_rank=rank
                        best_index=i
            if best_index==-1:
                break
            merged=tokens[best_index]+tokens[best_index+1]
            tokens=tokens[:best_index]+[merged]+tokens[best_index+2:]

        
        """     
        for pat in self.merges:
             tokens=merge_word(tokens,pat) 逻辑有误"""
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                raise KeyError
        return ids
    def encode(self,text:str)->list[int]:
        if self.special_tokens:
            #按长度降序，防止重叠匹配出错
            sorted_specials=sorted(self.special_tokens,key=len,reverse=True)
            escaped='|'.join(re.escape(tok) for tok in sorted_specials)
            split_re=re.compile(f"({escaped})")
            segments=split_re.split(text)
        else:
            segments=[text]

        result=[]

        for segment in segments:
            if not segment:
                continue
            if segment in self.special_token_ids:
                result.append(self.special_token_ids[segment])
                continue
            for match in re.finditer(PAT,segment):
                word=match.group()
                word_bytes=word.encode("utf-8")
                if word_bytes in self.token_to_id:
                    result.append(self.token_to_id[word_bytes])
                else:
                    result.extend(self.encode_bytes(word_bytes))
        return result

    def encode_iterable(self,iterable:Iterable[str]) ->Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    def decode(self,ids:list[int]) ->str:
        token=b"".join((self.vocab[id]) for id in ids)
        decoded=token.decode("utf-8",errors="replace")
        return decoded
    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        #反序列化（使用pickle）
        with open(vocab_filepath,"rb") as f:
            vocab=pickle.load(f)
        with open(merges_filepath,"rb") as f1:
            merges=pickle.load(f1)
        return cls(vocab,merges,special_tokens)