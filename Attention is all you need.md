# Attention Is All You Need 阅读笔记

**论文**：Vaswani et al. "Attention Is All You Need." NeurIPS 2017.  
**意义**：提出了 Transformer 架构，奠定了 GPT、BERT 等大模型的基础。

---

## 1. 论文要解决什么问题？

在 Transformer 之前，处理序列数据（如文本、语音）主流用的是 **RNN/LSTM**。RNN 的问题是：
- **无法并行**：必须一个词一个词按顺序算，训练慢
- **长距离依赖**：句子太长时，前面的信息容易“遗忘”

论文提出了一种**完全基于 Attention 机制**的架构——**Transformer**，彻底抛弃了循环和卷积。

---

## 2. 核心方法

### 2.1 Self-Attention（自注意力）

**核心公式**：Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V

**解释**：
- **Q（Query）**：“我想查什么”
- **K（Key）**：“我有什么标签”
- **V（Value）**：“实际内容是什么”
- **Q·K^T**：计算每个位置对其他位置的“注意力分数”
- **/√d_k**：缩放，防止分数太大导致梯度消失
- **softmax**：把分数转成概率（和为1）
- **× V**：按权重加权求和，得到新表示

**通俗理解**：每个词都问“我应该更关注句子里的哪些其他词”，然后把那些词的信息汇总到自己身上。

### 2.2 Multi-Head Attention（多头注意力）

- 把 Q、K、V 分别投影到多个不同的空间，每个“头”独立计算注意力
- 不同头能关注不同的关系（比如一个头关注语法，另一个头关注指代）
- 最后把多个头的结果拼接起来



### 2.3 Position Encoding（位置编码）

- Attention 本身不区分词的顺序（“我爱你”和“你爱我”对它来说一样）
- 所以需要额外加入位置信息，告诉模型每个词在第几个位置
- 论文用的是**三角函数**编码

### 2.4 残差连接 + LayerNorm

- **残差连接**：`x = x + Attention(x)`，让梯度能直接传到浅层，解决梯度消失
- **LayerNorm**：每一层前做归一化，让训练更稳定

---

## 3. 整体架构

Transformer 分 **Encoder**（编码器）和 **Decoder**（解码器）两部分：

Encoder：输入 → Embedding + 位置编码 → N个Encoder Block → 输出
Decoder：输出 → Embedding + 位置编码 → N个Decoder Block → 预测下一个词

每个 Encoder Block：Self-Attention → 残差+LayerNorm → FFN → 残差+LayerNorm
每个 Decoder Block：Masked Self-Attention → 残差+LayerNorm → Cross-Attention → 残差+LayerNorm → FFN → 残差+LayerNorm

FFN（前馈网络）	两层全连接网络（第一层升维到4倍，第二层降维回来）。对每个位置独立做非线性变换，让模型能学到更复杂的特征

Cross-Attention	Decoder 的 Q 来自 Decoder 自己，K 和 V 来自 Encoder 的输出。作用是：Decoder 在生成下一个词时，可以“回头”看 Encoder 理解的输入句子，做翻译/对齐。比如生成法语的“Je”时，去看英语的“I”
## 4. 关键创新点总结

| 创新点 | 解决的问题 |
|--------|-----------|
| **Self-Attention** | 一步到位捕捉全局依赖，替代 RNN 的逐步计算 |
| **Multi-Head Attention** | 让模型从多个角度关注信息 |
| **位置编码** | 让 Attention 感知顺序 |
| **残差连接 + LayerNorm** | 让深层网络能训练（解决梯度消失） |
| **抛弃 RNN/CNN** | 实现并行计算，大幅提升训练速度 |