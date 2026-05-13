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

    TODO:
    请在 forward 中完成：
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

        # TODO 1:
        # 如果 neg_emb 不为 None，将 pos_emb 和 neg_emb 在 batch 维度拼接；
        # 否则只使用 pos_emb 作为 passages。
        #
        # 提示：
        # passages 的形状应为 [B, D] 或 [2B, D]
        passages = None

        # TODO 2:
        # 计算 q_emb 和 passages 的相似度矩阵。
        #
        # 提示：
        # q_emb shape:    [B, D]
        # passages shape: [N, D]
        # logits shape:   [B, N]
        # 注意除以 temperature。
        logits = None

        # TODO 3:
        # 构造 labels。
        #
        # 第 i 个 query 的正确 passage 是第 i 个 positive，
        # 所以 labels 应该是 [0, 1, 2, ..., B-1]。
        labels = None

        # TODO 4:
        # 使用 F.cross_entropy 计算 loss。
        loss = None

        return loss, logits