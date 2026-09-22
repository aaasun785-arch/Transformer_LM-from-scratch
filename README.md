# Transformer-LM：from scratch

从零实现并训练一个 Decoder-only Transformer 语言模型，覆盖文本分词、数据预处理、模型构建、训练验证、实验追踪和文本生成。
使用 PyTorch 实现，BPE Tokenizer、AdamW 优化器、causal multi-head attention和训练流程均从基础模块开始构建，主要在 TinyStories 数据集上进行训练和实验。

基于 [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) Assignment 1 。

## 项目概览

Transformer-LM 实现了从原始文本到生成文本的完整语言模型训练流程：

原始文本-> Tokenizer -> 二进制数据集 -> Transformer训练  -> 文本生成

不调用现成的 Transformer 模型，主要部分从零开始构建，同时建立一套可复现的训练与实验流程。

## 主要实现

模型组件：

- Linear 与 Embedding；
- RMSNorm；
- RoPE；
- Causal Multi-Head Self-Attention；
- SwiGLU ；
- Pre-Norm Transformer Block；
- Decoder-only Transformer Language Model。

训练组件：

- Byte-level BPE Tokenizer；
- Cross-Entropy Loss；
- AdamW 优化器；
- Learning Rate Schedule；
- Gradient Clipping；
- 基于 `np.memmap` 的数据加载；
- 定期验证与指标记录；
- Weights & Biases 实验追踪。

生成组件：

- Temperature Sampling；
- Top-p Sampling；


## 实验结果
- 比较了不同学习率对训练稳定性和收敛速度的影响；
- 比较了不同模型规模下的验证损失、训练速度；
- Temperature 和 Top-p 对生成文本的影响
<p align="center">
  <img src="docs/figures/屏幕截图 2026-09-22 211131.png"
       alt="不同learning_rate实验对比"
       width="760">
</p>
<p align="center">
  <img src="docs/figures/屏幕截图 2026-09-22 210744.png"
       alt="不同d_model实验对比"
       width="760">
</p>
<p align="center">
  <img src="docs/figures/屏幕截图 2026-09-22 211349.png"
       alt="不同num_layers实验对比"
       width="760">
</p>

固定其他参数，在不同temperature和top-p下生成文本结果在："docs/output-text.md"

示例：
```text
===== Generated Text =====
temperature=0.8 top_p=0.9

Once upon a time, there was a little girl named Lily. She had a pet bird named Blue. Blue was very small and loved to fly with Lily. They played together every day.
One day, Lily saw a big red ball in the yard. She wanted to play with it. Blue flew down to Lily and said, "Can I play with the big red ball?" Lily smiled and said, "Yes, let's play together!"
Lily and Blue played with the big red ball. They had so much fun. After a while, they got tired and decided to rest. Lily said, "Thank you, Blue, for playing with me and the big red ball." Blue smiled and said, "You're welcome, Lily. We had fun today."
From that day on, Lily and Blue played together every day. They were very happy. And Lily learned that sharing her love and friendship with friends is more important than being small.
<|endoftext|>
```text

## 运行

项目使用 `uv` 管理依赖。

```bash
git clone https://github.com/USERNAME/ground-up-lm.git
cd ground-up-lm
uv sync
```
