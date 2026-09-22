import time
import numpy as np
import torch
import wandb
from cs336_basics.transformer import (
    TransformerLM,AdamW,cross_entropy,
    gradient_clipping,learning_rate_schedule,save_checkpoint
)

# ===== 配置 =====
TRAIN_PATH="data/tinystories_train.bin"
VAL_PATH="data/tinystories_valid.bin"
DEVICE="cuda"

VOCAB_SIZE=10000
CONTEXT_LENGTH=256
D_MODEL=512
NUM_LAYERS=4
NUM_HEADS=8
D_FF=1344
ROPE_THETA=10000.0

BATCH_SIZE=8
MAX_ITERS=500
LR_MAX=1e-3
LR_MIN=1e-4
WARMUP_ITERS=50
EVAL_INTERVAL=50

def get_batch(data):
    starts=np.random.randint(0,len(data)-CONTEXT_LENGTH-1,size=BATCH_SIZE)
    x=np.stack([data[i:i+CONTEXT_LENGTH] for i in starts])
    y=np.stack([data[i+1:i+CONTEXT_LENGTH+1] for i in starts])
    x=torch.tensor(x,dtype=torch.long,device=DEVICE)
    y=torch.tensor(y,dtype=torch.long,device=DEVICE)
    return x,y

@torch.no_grad()
def evaluate(model,val_data,eval_iters=10):
    model.eval()
    losses=[]
    for _ in range(eval_iters):
        x,y=get_batch(val_data)
        loss=cross_entropy(model(x),y)
        losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    run=wandb.init(
        project="cs336-tinystories",
        name=f"lr_{LR_MAX}",
        config={
            "learning_rate":LR_MAX,
            "batch_size":BATCH_SIZE,
            "max_iters":MAX_ITERS,
            "context_length":CONTEXT_LENGTH,
            "d_model":D_MODEL,
            "num_layers":NUM_LAYERS,
            "num_heads":NUM_HEADS,
            "d_ff":D_FF,
        }
    )

    # memmap 不会一次把整个数据集加载进内存
    train_data=np.memmap(TRAIN_PATH,dtype=np.uint16,mode="r")
    val_data=np.memmap(VAL_PATH,dtype=np.uint16,mode="r")
    print(f"train tokens: {len(train_data):,}")
    print(f"val tokens:   {len(val_data):,}")

    model=TransformerLM(
        VOCAB_SIZE,CONTEXT_LENGTH,D_MODEL,
        NUM_LAYERS,NUM_HEADS,D_FF,ROPE_THETA
    ).to(DEVICE)

    optimizer=AdamW(model.parameters(),lr=LR_MAX)
    start=time.time()

    for step in range(MAX_ITERS):
        # cosine learning rate
        lr=learning_rate_schedule(
            step,LR_MAX,LR_MIN,
            WARMUP_ITERS,MAX_ITERS
        )
        for group in optimizer.param_groups:
            group["lr"]=lr

        x,y=get_batch(train_data)
        logits=model(x)
        loss=cross_entropy(logits,y)

        if not torch.isfinite(loss):
            print(f"diverged at step {step}")
            wandb.log({"diverged":1},step=step)
            break

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(),1.0)
        optimizer.step()

        # 每10步记录训练 loss、lr、时间
        if step%10==0:
            elapsed=time.time()-start
            print(f"step={step:4d} train_loss={loss.item():.4f} lr={lr:.2e}")
            wandb.log({
                "train_loss":loss.item(),
                "learning_rate":lr,
                "elapsed_time":elapsed
            },step=step)

        if step%EVAL_INTERVAL==0:
            val_loss=evaluate(model,val_data)
            print(f"         val_loss={val_loss:.4f}")
            wandb.log({"val_loss":val_loss},step=step)

    save_checkpoint(model,optimizer,step,"checkpoint.pt")
    wandb.finish()
    print(f"finished, time={time.time()-start:.1f}s")

if __name__=="__main__":
    main()