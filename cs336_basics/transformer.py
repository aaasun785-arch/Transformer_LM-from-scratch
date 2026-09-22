from torch import nn
import torch
import math


"""---------------LAYERS--------------- """

class Linear(nn.Module):
    def __init__(self, in_features:int,
                  out_features:int,
                  device=None,
                 dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))#pytorch里面的w就是w.T
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight,mean=0.0,std=std,a=-3 * std,b=3 * std)
    def forward(self,
                x:torch.Tensor)->torch.Tensor:
        return torch.matmul(x,self.weight.T)

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings:int,
                embedding_dim:int,
                device=None, dtype=None):
        super().__init__()
        self.embedding=nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.embedding,mean=0.0,std=1.0,a=-3,b=3)
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.embedding[token_ids]#通过矩阵indexing，把token_ids中的每个位置改成对应embedding中的数据，并且output维数加一

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        out = x / rms * self.weight
        return out.to(in_dtype)
        

class SwiGLU(nn.Module):
    def __init__(self,d_model: int,
                 d_ff: int,
                 device=None,
                 dtype=None):
        super().__init__()
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)
    def forward(self, x):
        gate = self.w1(x)
        value = self.w3(x)
        return self.w2(torch.nn.functional.silu(gate) * value)

    

""" ---------------ATTENTION--------------- """

class RoPE(nn.Module):
    def __init__(self,
                 theta: float, 
                 d_k: int, 
                 max_seq_len: int, 
                 device=None):
        super().__init__()
        #进行断言
        assert d_k%2==0,"d_k must be even"
        self.theta=theta
        self.d_k=d_k
        self.max_seq_len=max_seq_len
        angle_k=1.0/(theta**((torch.arange(0,d_k,step=2,device=device).float())/d_k))
        self.register_buffer("angle_k",angle_k,persistent=False)
        positions=torch.arange(max_seq_len,device=device).float()
        angles=torch.outer(positions,angle_k)
        #外积：angles[i,j]=postions[i]*angle_k[j]
        self.register_buffer("sin",torch.sin(angles),persistent=False)
        self.register_buffer("cos",torch.cos(angles),persistent=False)  
    def forward(self,
                x:torch.Tensor,
                token_positions:torch.Tensor)->torch.Tensor:
        token_positions=token_positions.to(x.device)
        cos=self.cos.to(device=x.device,dtype=x.dtype)[token_positions]
        sin=self.sin.to(device=x.device,dtype=x.dtype)[token_positions]

        x1=x[...,0::2]
        x2=x[...,1::2]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1,out2],dim=-1).flatten(-2)#竖过来放，然后再展平
        """ torch.concat([out1.unsqueeze(-1),out2.unsqueeze(-1)],dim=1).flatten(-2) """

def softmax(i:int,
            x:torch.Tensor)->torch.Tensor:
    x0_max=x.max(dim=i,keepdim=True).values
    
    entry=(x-x0_max).exp()
    return entry/entry.sum(dim=i,keepdim=True)
def scaled_dot_product_attention(query:torch.Tensor,
                                 key:torch.Tensor,
                                 value:torch.Tensor,
                                 mask=None
                                 )->torch.Tensor:
    d_k=query.shape[-1]
    scores=(query@(key.transpose(-1,-2)))/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(~mask,float("-inf"))
    attention=softmax(-1,scores)
    return attention@value#整个序列对key做归一化

def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key,
    token_positions,
) :
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope=RoPE(theta,d_k,max_seq_len)
    return rope(in_query_or_key,token_positions)

def run_scaled_dot_product_attention(
    Q,
    K,
    V,
    mask = None,
) :
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q,K,V,mask)

from einops import rearrange
class multihead_self_attention(nn.Module):
    def __init__(self,d_model:int,
                 num_heads:int,
                 theta,
                 seq):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.seq = seq
        self.theta = theta
        self.q_proj = Linear(d_model,d_model)
        self.k_proj = Linear(d_model,d_model)
        self.v_proj = Linear(d_model,d_model)
        self.output_proj = Linear(d_model,d_model)
    def forward(self,x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q,"... seq (h d) -> ... h seq d",h=self.num_heads)
        k = rearrange(k,"... seq (h d) -> ... h seq d",h=self.num_heads)
        v = rearrange(v,"... seq (h d) -> ... h seq d",h=self.num_heads)
        seq_len=x.shape[-2]
        positions=torch.arange(seq_len,device=x.device)
        q=run_rope(self.d_k,self.theta,self.seq,q,positions)
        k=run_rope(self.d_k,self.theta,self.seq,k,positions)
        mask=torch.tril(
            torch.ones(seq_len,seq_len,device=x.device)).bool()
        out=scaled_dot_product_attention(q,k,v,mask)
        out=rearrange(out,"... h seq d -> ... seq (h d)")
        return self.output_proj(out)



""" ---------------TRANSFORMER--------------- """

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,seq_len):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = multihead_self_attention(d_model,num_heads,theta,seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model,d_ff)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(self,vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,num_heads,d_ff,rope_theta,context_length)
            for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size,bias=False)
    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
    
import math
from typing import Optional,Callable 
def cross_entropy(logits:torch.Tensor,
                  target:torch.Tensor):
    shifted_logits=logits-logits.max(dim=-1,keepdim=True).values
    logsum=torch.log(torch.exp(shifted_logits).sum(dim=-1))
    target_logits=shifted_logits.gather(dim=-1,index=target.unsqueeze(-1)).squeeze(-1)
    return (logsum-target_logits).mean()#对整个batch进行平均，方便进行反向传播的计算
class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 eps=1e-8,
                 betas=(0.9,0.999),
                 weight_decay=0.01):
        beta1,beta2=betas
        defaults = {"lr": lr,
                    "eps":eps,
                    "betas":betas,
                    "weight_decay":weight_decay,
                    }
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:#不同的参数可以有不同的超参数
            lr = group["lr"]  
            eps=group["eps"]
            weight_decay=group["weight_decay"]
            beta1,beta2=group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad=p.grad
                state=self.state[p]
                #第一次遇到参数进行初始化
                if len(state)==0:
                    state["t"]=0
                    state["m"]=torch.zeros_like(p)
                    state["v"]=torch.zeros_like(p)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                adjusted_lr = (lr* math.sqrt(1 - beta2**t)/ (1 - beta1**t))
                p.data.mul_(1 - weight_decay * lr)

                p.data.addcdiv_(
                    m,
                    torch.sqrt(v) + eps,
                    value=-adjusted_lr,
                )
        return loss
def learning_rate_schedule(t,lr_max,lr_min,Tw,Tc):
    progress=(t-Tw)/(Tc-Tw)
    if t <Tw:
        return (t/Tw)*lr_max
    elif t>=Tw and t<=Tc:
        return lr_min+0.5*(1+math.cos(progress*math.pi))*(lr_max-lr_min)
    elif t>Tc:
        return lr_min
def gradient_clipping(parameters,
                      l2_norm:float):
    eps=1e-6
    total_norm=0.0
    params=list(parameters)#防止给的params是generator
    for p in params:
        if p.grad is None:
            continue
        total_norm+=torch.sum(p.grad**2)
    total_norm=total_norm.sqrt()
    if total_norm>l2_norm:
        scale=l2_norm/(total_norm+eps)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scale)

import numpy as np
def batching(x:np.array,
                 batch_size:int,
                 context_length:int,
                 device:str)->tuple[torch.Tensor,torch.Tensor]:
    starts=np.random.randint(0,len(x)-context_length,batch_size)
    inputs=np.stack([x[i:i+context_length] for i in starts])
    targets=np.stack([x[i+1:i+1+context_length] for i in starts])
    inputs=torch.tensor(inputs,device=device)
    targets=torch.tensor(targets,device=device)
    return inputs,targets

def save_checkpoint(model, optimizer, iteration, out):
    checkpoints={"model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "iteration":iteration}
    torch.save(checkpoints,out)
def load_checkpoint(src, model:nn.Module, optimizer:torch.optim.Optimizer):
    checkpoints=torch.load(src)
    model.load_state_dict(checkpoints["model"])
    optimizer.load_state_dict(checkpoints["optimizer"])
    return checkpoints["iteration"]

""" 实现整个模块 """
""" import time
import wandb
run=wandb.init(
    project="transformer-experiment",
    name="baseline",
    config={
        "learning_rate": 1e-3,
        "batch_size": 32,
        "max_steps": 5000
    },
    reinit=True

)
start_time=time.time()
def training_together(train_path,
    val_path,
    checkpoint_path,
    vocab_size=10000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=8,
    d_ff=1344,
    batch_size=32,
    max_iters=10000,
    lr=3e-4,
    eval_interval=100,
    save_interval=1000,
    device="cuda"):
    #加载数据
    train_data=np.load(train_path,mmap_mode="r")
    val_data=np.load(val_path)
    #创建模型
    model=TransformerLM(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta=0.001).to(device)
    #optimizer&loss function
    optimizer=AdamW(model.state_dict())
    #循环
    for iteration in range(max_iters):
        x,y=batching(train_data,batch_size,context_length,device)
        logits=model(x)
        loss=cross_entropy(logits,y)

        optimizer.zero_grad()
        loss.backward()
        #gradient clipping
        gradient_clipping(model,1.0)
        optimizer.step()
        #log
        if iteration%10==0:
            print(f"iter:{iteration} ")
            print(f"training_loss:{loss.item():.4f}")
        if iteration%eval_interval==0:
            model.eval()
            with torch.inference_mode():
                x_val,y_val=batching(val_data,batch_size,context_length,device)
                logits_val=model(x_val)
                loss_val=cross_entropy(logits_val,y_val)
                elapsed_time = time.time() - start_time
                run.log(
                {
                "train_loss": loss.item(),
                "val_loss": loss_val,
                "elapsed_time": elapsed_time,
                },
                step=iteration
            )   
                model.train()
        #checkpoint
        if iteration % save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                iteration,
                checkpoint_path,
            )
    run.finish() """


def apply_top_p(probs,top_p):
    sorted_probs,sorted_indices=torch.sort(sorted_probs,dim=1,descending=True)
    cumsum_probs=torch.cumsum(probs,-1)
    sorted_mask=(cumsum_probs-sorted_probs)>top_p
    sorted_probs=torch.masked_fill(sorted_probs,sorted_mask,0.0)
    sorted_probs=sorted_probs/sorted_probs.sum(dim=-1,keepdim=True)
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_probs
    )
    return filtered_probs

@torch.no_grad()
def decode(model:nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>"):
    model.eval()
    device=next(model.parameters()).device
    context_length:int=256,
    token_ids=tokenizer.encode(prompt)
    tokens=torch.tensor(token_ids,dtype=torch.long,device=device).unsqueeze(0)
    eos_token_id=tokenizer.encode(eos_token)[0]

    for _ in range(max_new_tokens):
        if "has max_length":
            model_input=tokens[:,-context_length]
        else:
            model_input=tokens
        logits=model(model_input)
        next_token_logits=logits[:,-1,:]
        if temperature==0:
            next_token=torch.argmax(next_token_logits,dim=-1,keepdim=True)
        else:
            if temperature<0:
                raise ValueError
            next_token_logits=next_token_logits/temperature    
            probs=softmax(-1,next_token_logits)
            if top_p<1.0:
                probs=apply_top_p(probs,top_p)
            next_token=torch.multinomial(probs,1)
        tokens=torch.concat([tokens,next_token],dim=-1)
        if next_token.item()==eos_token:
            break
    output0=tokens.squeeze(0).tolist()
    output=tokenizer.decode(output0)
    return output
