"""V2.1: topology-aligned structural encoding before diagonal Q/K/V.

All pair matrices use [receiver, source]. Topological order is a display and
execution convention, never a positional feature. Parameters are independent of
node count. ``sparse=False`` is the ablation: keep sorting and fixed aggregation,
replace only the learned sparse transform with the identity.
"""
from __future__ import annotations

import heapq

import torch
from torch import nn

from .. import functional as dsl
from .diagonal import DiagonalRelationBlock
from .encoding import FEATURE_COLUMNS


class StructureParameterDefinition(nn.Module):
    """Own structural gains and translate graph metadata into aligned matrices.

    Input is the batch_graphs dictionary. x: [B,N,F], valid: [B,N],
    allowed/distances: [B,N,N], relations: [B,N,N,R]. ``order[b,k]`` is
    the original index of sorted node k; inverse_order undoes that permutation.
    Neither sorting nor fixed dependency matrices have trainable parameters.
    """

    def __init__(self, sparse: bool = True):
        super().__init__()
        self.sparse = sparse
        if sparse:
            self.self_scale = nn.Parameter(torch.tensor(1.0))
            self.predecessor_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('self_scale', torch.tensor(1.0))
            self.register_buffer('predecessor_scale', torch.tensor(0.0))

    @staticmethod
    def direct_dependencies(batch):
        """Boolean [B,N,N]; shortest distance 1 identifies a direct edge."""
        valid = batch['valid']
        n = valid.shape[1]
        eye = torch.eye(n, dtype=torch.bool, device=valid.device)[None]
        return ((batch['distances'] == 1) & batch['allowed'] & ~eye
                & valid[:, :, None] & valid[:, None, :])

    @staticmethod
    def reorder_nodes(value, order):
        """Apply each graph's own permutation along node axis 1."""
        shape = (*order.shape, *((1,) * (value.ndim - 2)))
        return value.gather(1, order.reshape(shape).expand_as(value))

    @classmethod
    def reorder_pairs(cls, value, order):
        """Synchronously permute receiver and source axes, preserving ports."""
        value = cls.reorder_nodes(value, order)
        shape = (order.shape[0], 1, order.shape[1], *((1,) * (value.ndim - 3)))
        return value.gather(2, order.reshape(shape).expand_as(value))

    @classmethod
    def prepare(cls, batch):
        """Kahn topological sort; reject cycles, place padding last.

        CPU traversal is metadata preparation, not a differentiable operation.
        Independent branches may tie; their index tie-break has no effect on
        predictions because no node-index/bandwidth parameters are learned.
        """
        valid = batch['valid']
        if valid.ndim != 2 or valid.shape[1] == 0:
            raise ValueError('Expected nonempty padded node axis [B,N]')
        direct = cls.direct_dependencies(batch)
        orders = []
        for graph, mask in zip(direct.detach().cpu(), valid.detach().cpu()):
            nodes = mask.nonzero(as_tuple=True)[0].tolist()
            if not nodes:
                raise ValueError('Each graph needs at least one valid node')
            degree = graph.sum(-1).tolist()
            ready = [i for i in nodes if degree[i] == 0]
            heapq.heapify(ready)
            order = []
            successors = graph.transpose(0, 1)
            while ready:
                source = heapq.heappop(ready)
                order.append(source)
                for target in successors[source].nonzero(as_tuple=True)[0].tolist():
                    degree[target] -= 1
                    if degree[target] == 0:
                        heapq.heappush(ready, target)
            if len(order) != len(nodes):
                raise ValueError('Topological encoding requires an acyclic dependency graph')
            orders.append(order + (~mask).nonzero(as_tuple=True)[0].tolist())
        order = torch.tensor(orders, device=valid.device, dtype=torch.long)
        # Explorer STEP 02: 拓扑排序后，节点轴与成对矩阵的两轴同步重排。
        # LaTeX: $$X=\Pi X_{raw},\quad B=\Pi B_{raw}\Pi^\top\ (B\in\{E,M,D\})$$
        aligned = dict(batch)
        for key in ('x', 'valid'):
            aligned[key] = cls.reorder_nodes(batch[key], order)
        for key in ('allowed', 'distances', 'relations'):
            aligned[key] = cls.reorder_pairs(batch[key], order)
        return aligned, order, order.argsort(dim=1)

    @classmethod
    def fixed_matrices(cls, batch, dtype):
        """I on valid nodes and P = row-normalized direct predecessors.

        P has a zero diagonal; roots have a zero row. Residual addition supplies
        self information exactly once during the fixed aggregation stage.
        """
        valid = batch['valid']
        direct = cls.direct_dependencies(batch).to(dtype)
        predecessors = direct / direct.sum(-1, keepdim=True).clamp_min(1)
        identity = torch.diag_embed(valid.to(dtype))
        return identity, predecessors

    def matrices(self, batch, dtype):
        identity, predecessors = self.fixed_matrices(batch, dtype)
        sparse = self.self_scale.to(dtype) * identity + self.predecessor_scale.to(dtype) * predecessors
        return sparse, predecessors


class SparseStructureEncoder(nn.Module):
    """[B,N,D] -> [B,N,D], with S = a I + b P, then (I + P) S H.

    a=1,b=0 initially. Gains may become signed. S is structurally sparse but is
    materialized as a dense batched tensor for the current small-graph backend.
    """

    def __init__(self, sparse: bool = True):
        super().__init__()
        self.p = StructureParameterDefinition(sparse)

    def forward(self, h, sparse_matrix, predecessors):
        # Explorer STEP 04: 结构变换：P 为归一化直接前驱，a、b 为可学习标量。
        # LaTeX: $$P_{ij}=E_{ij}/\max(1,\sum_k E_{ik}),\quad S=aI+bP,\quad L=SH_0$$
        local = dsl.mix(h, sparse_matrix)
        # Explorer STEP 05: 固定前驱聚合，加回自身表示。
        # LaTeX: $$H_s=L+PL$$
        encoded = dsl.add(local, dsl.mix(local, predecessors))
        return encoded, local


class StructuredGraphScoreModel(nn.Module):
    """Topology -> projection -> sparse/fixed encoding -> diagonal relation blocks.

    Defaults: feature width 32, two independent relation blocks. Train a separate
    model per task; there is no task embedding or required task ID in the batch.
    ``objective='normalized_score'`` returns one sigmoid score [B] and owns no
    completion head. The default ``legacy`` mode retains the historical raw
    score/completion_logit outputs for existing relation-v2 experiments.
    With return_relations=True, return
    sorted-node states, order/inverse_order, structural matrices/states, and
    each block's raw similarity, weighted similarity, softmax relation and V.
    Use node_order to align node IDs; use inverse_order to restore tensors.
    """

    def __init__(self, width=32, layers=2, sparse=True, objective='legacy'):
        super().__init__()
        if objective not in ('legacy', 'normalized_score'):
            raise ValueError('objective must be legacy or normalized_score')
        self.config = dict(width=width, layers=layers, sparse=sparse, objective=objective)
        self.objective = objective
        self.input = nn.Linear(len(FEATURE_COLUMNS), width)
        self.blocks = nn.ModuleList([DiagonalRelationBlock(width) for _ in range(layers)])
        # Masked mean, fixed-scale sum, and log node count: 2 * width + 1.
        self.readout = nn.Sequential(nn.Linear(2 * width + 1, width), nn.GELU(),
                                     nn.Linear(width, width), nn.GELU())
        self.score = nn.Linear(width, 1)
        if objective == 'legacy':
            self.completion = nn.Linear(width, 1)
        self.encoder = SparseStructureEncoder(sparse=sparse)

    def forward(self, batch, return_relations=False):
        aligned, order, inverse = self.encoder.p.prepare(batch)
        # Explorer STEP 03: 共享线性投影：每个节点从 46 维映射为 32 维。
        # LaTeX: $$H_0=XW_{in}^{\top}+b_{in}$$
        h = self.input(aligned['x'])
        sparse, predecessors = self.encoder.p.matrices(aligned, h.dtype)
        projected = h
        h, local = self.encoder(h, sparse, predecessors)
        structural = h
        relations = []
        for block in self.blocks:
            h, debug = block(h, aligned, return_relations)
            if debug is not None:
                relations.append(debug)
        # Explorer STEP 16: 只汇聚有效节点，拼接均值、固定缩放求和与节点数量。
        # LaTeX: $$n=\sum_i v_i,\quad t=\sum_i v_i H_i,\quad z=[t/\max(n,1);\ t/32;\ \ln(1+\max(n,1))/5]$$
        valid = aligned['valid'].unsqueeze(-1)
        count = valid.sum(1).clamp_min(1)
        total = (h * valid).sum(1)
        features = torch.cat((total / count, total / 32,
                              count.float().log1p() / 5), dim=-1)
        # Explorer STEP 17: 图级 MLP：65 → 32 → 32。
        # LaTeX: $$g=\operatorname{GELU}(\operatorname{GELU}(zW_1^{\top}+b_1)W_2^{\top}+b_2)$$
        pooled = self.readout(features)
        # Explorer STEP 18: normalized_score 模式返回单一 sigmoid 分数；标签不参与前向。
        # LaTeX: $$\ell=gw+b,\quad \widehat{s}=\sigma(\ell)$$
        score = self.score(pooled).squeeze(-1)
        if self.objective == 'normalized_score':
            result = {'score': score.sigmoid()}
        else:
            result = {'score': score,
                      'completion_logit': self.completion(pooled).squeeze(-1)}
        if return_relations:
            result.update(node_order=order, inverse_order=inverse,
                          node_states=h, relations=relations,
                          structure={'sparse_matrix': sparse, 'predecessors': predecessors,
                                     'projected': projected, 'after_sparse': local,
                                     'after_dependency': structural,
                                     'self_scale': self.encoder.p.self_scale,
                                     'predecessor_scale': self.encoder.p.predecessor_scale,
                                     'valid': aligned['valid']})
        return result
