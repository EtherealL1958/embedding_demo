import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveRetrievalLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss / InfoNCE。

    输入：
    q_emb:   [B, D]
    pos_emb: [B, D]
    neg_emb: [B, D] 或 None

    passage 矩阵应构造为：
    [pos_0, pos_1, ..., pos_B-1, neg_0, neg_1, ..., neg_B-1]

    第 i 个 query 的正确 passage 是第 i 个 positive。
    因此 labels = [0, 1, 2, ..., B-1]

    forward 完成：
    1. 拼接 positive passages 和 negative passages；
    2. 计算 query 与所有 passages 的相似度矩阵；
    3. 构造 labels；
    4. 使用 cross entropy 计算 InfoNCE loss。
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, q_emb, pos_emb, neg_emb=None):
        """
        参数：
            q_emb:   [B, D]，query embedding
            pos_emb: [B, D]，positive passage embedding
            neg_emb: [B, D]，negative passage embedding，可以为 None

        返回：
            loss:   对比学习损失
            logits: [B, num_passages]，query 和 passage 的相似度矩阵
        """

        if neg_emb is not None:
            passages = torch.cat([pos_emb, neg_emb], dim=0)
        else:
            passages = pos_emb

        logits = torch.matmul(q_emb, passages.t()) / self.temperature

        labels = torch.arange(q_emb.size(0), device=q_emb.device)

        loss = F.cross_entropy(logits, labels)

        return loss, logits
