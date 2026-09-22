"""V2: diagonal Q/K/V and positive, structural-type-shared pair scaling.

Pair matrices are indexed [receiver, source]. No node-index-specific parameters
or learned dense Q/K/V projections are used. Exported weights describe routing,
not causal contributions to task loss.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import FEATURE_COLUMNS

RELATION_TYPES = ('self', 'direct_predecessor', 'other_ancestor')


class DiagonalParameterDefinition(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.wq = nn.Parameter(torch.ones(width))
        self.wk = nn.Parameter(torch.ones(width))
        self.wv = nn.Parameter(torch.ones(width))
        # softplus(inverse_softplus(1)) = 1 initially.
        self.raw_pair_scale = nn.Parameter(torch.full((3,), math.log(math.expm1(1.))))

    def pair_scales(self):
        return F.softplus(self.raw_pair_scale).clamp_min(1e-6)


class DiagonalRelationBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.p = DiagonalParameterDefinition(width)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 2 * width),
            nn.GELU(), nn.Linear(2 * width, width),
        )

    def forward(self, h, batch, return_relations=False):
        # h [B,N,D]; all node-count-dependent tensors come from graph structure.
        # Explorer STEP 06 / 11: 两层分别执行节点归一化，各层参数独立。
        # LaTeX: $$Z=\operatorname{LayerNorm}(H)$$
        z = self.norm(h)
        # Explorer STEP 07 / 12: 按通道逐元素缩放；V 在关系矩阵计算后求值。
        # LaTeX: $$Q=Z\odot w_q,\quad K=Z\odot w_k,\quad V=Z\odot w_v$$
        q, k = z * self.p.wq, z * self.p.wk
        # Explorer STEP 08 / 13: 结构类型缩放与祖先掩码；不可达位置设为负无穷。补齐行保留自身。
        # LaTeX: $$T=QK^{\top}/\sqrt{d},\quad C_{ij}=\max(\operatorname{softplus}(r_{type(i,j)}),10^{-6}),\quad A=\operatorname{softmax}_{j}(\operatorname{mask}_{M}(T\odot C))$$
        similarity = q @ k.transpose(-1, -2) / math.sqrt(h.shape[-1])
        n = h.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=h.device)[None]
        kind = torch.where(batch['distances'] == 1, 1, 2)
        kind = torch.where(eye, 0, kind)
        scales = self.p.pair_scales()
        pair_scale = scales[kind]
        weighted = similarity * pair_scale
        # Padding rows retain a self entry to avoid all-masked softmax; padding
        # cannot send messages to valid nodes and is excluded from readout.
        allowed = batch['allowed'] & batch['valid'][:, None, :]
        allowed = allowed | (eye & ~batch['valid'][:, :, None])
        logits = weighted.masked_fill(~allowed, -torch.inf)
        relation = logits.softmax(dim=-1)
        v = z * self.p.wv
        # Explorer STEP 09 / 14: 聚合消息并加上本层原始输入的残差。
        # LaTeX: $$U=H+AV$$
        message = relation @ v
        h = h + message
        # Explorer STEP 10 / 15: 先归一化，再扩展通道、GELU、回投影，最后加回 U。
        # LaTeX: $$H^{\prime}=U+\operatorname{GELU}(\operatorname{LayerNorm}(U)W_1^{\top}+b_1)W_2^{\top}+b_2$$
        h = h + self.ffn(h)
        debug = None
        if return_relations:
            debug = {'similarity': similarity, 'pair_scale': pair_scale,
                     'weighted_similarity': weighted, 'relation': relation,
                     'allowed': allowed, 'relation_type': kind,
                     'value': v, 'message': message,
                     'wq': self.p.wq, 'wk': self.p.wk, 'wv': self.p.wv,
                     'qk_channel_product': self.p.wq * self.p.wk,
                     'type_scales': scales}
        return h, debug


class DiagonalGraphScoreModel(nn.Module):
    def __init__(self, width=32, layers=2, tasks=1):
        super().__init__()
        self.config = dict(width=width, layers=layers, tasks=tasks)
        self.input = nn.Linear(len(FEATURE_COLUMNS), width)
        self.blocks = nn.ModuleList([DiagonalRelationBlock(width) for _ in range(layers)])
        # Match V1's pooling and readout so the comparison changes the blocks.
        self.task = nn.Embedding(tasks, 16)
        self.readout = nn.Sequential(nn.Linear(2 * width + 17, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.score = nn.Linear(width, 1)
        self.completion = nn.Linear(width, 1)

    def forward(self, batch, return_relations=False):
        h = self.input(batch['x'])
        relations = []
        for block in self.blocks:
            h, debug = block(h, batch, return_relations)
            if debug is not None:
                relations.append(debug)
        valid = batch['valid'].unsqueeze(-1)
        count = valid.sum(1).clamp_min(1)
        total = (h * valid).sum(1)
        features = torch.cat((total / count, total / 32, self.task(batch['task']),
                              count.float().log1p() / 5), dim=-1)
        pooled = self.readout(features)
        result = {'score': self.score(pooled).squeeze(-1),
                  'completion_logit': self.completion(pooled).squeeze(-1)}
        if return_relations:
            result['relations'] = relations
            result['node_states'] = h
        return result
