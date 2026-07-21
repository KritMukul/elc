"""
ST-GCN style encoder for skeleton/mocap-marker gait sequences.

Input: [B, T, V, C] (batch, frames, joints/markers, xyz)
The graph adjacency is built as a simple fully-connected-within-radius graph
by default since exact anatomical marker connectivity depends on the mocap
marker set used (Skeletal_info.mat) - swap in the true bone graph there for
best results.
"""

import numpy as np
import torch
import torch.nn as nn


def build_default_adjacency(n_joints):
    """Identity + fully connected normalized adjacency (fallback graph)."""
    A = np.ones((n_joints, n_joints), dtype=np.float32)
    A = A / A.sum(axis=1, keepdims=True)
    return torch.from_numpy(A)


class GraphConv(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.register_buffer("A", A)
        self.linear = nn.Linear(in_c, out_c)

    def forward(self, x):
        # x: [B, T, V, C]
        x = torch.einsum("vw,btwc->btvc", self.A, x)
        return self.linear(x)


class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A, temporal_kernel=9, stride=1, dropout=0.3):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, A)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(dropout),
        )
        self.residual = (in_c == out_c and stride == 1)
        if not self.residual:
            self.res_conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=(stride, 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, T, V]
        res = x if self.residual else self.res_conv(x)
        h = self.gcn(x.permute(0, 2, 3, 1))          # -> [B, T, V, C]
        h = h.permute(0, 3, 1, 2)                     # -> [B, C, T, V]
        h = self.tcn(h)
        return self.act(h + res)


class STGCN(nn.Module):
    def __init__(self, n_joints, in_channels=3, hidden=(64, 128, 256),
                 out_dim=512, dropout=0.3, n_classes=2, adjacency=None):
        super().__init__()
        A = adjacency if adjacency is not None else build_default_adjacency(n_joints)
        self.data_bn = nn.BatchNorm1d(in_channels * n_joints)

        chans = [in_channels] + list(hidden)
        blocks = []
        for i in range(len(hidden)):
            stride = 2 if i > 0 else 1
            blocks.append(STGCNBlock(chans[i], chans[i + 1], A, stride=stride, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(hidden[-1], out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward_features(self, x):
        # x: [B, T, V, C] -> normalize -> [B, C, T, V]
        b, t, v, c = x.shape
        x = x.permute(0, 3, 2, 1).contiguous()          # [B, C, V, T]
        x = x.view(b, c * v, t)
        x = self.data_bn(x)
        x = x.view(b, c, v, t).permute(0, 1, 3, 2)       # [B, C, T, V]

        for blk in self.blocks:
            x = blk(x)

        x = self.pool(x).view(b, -1)   # [B, hidden[-1]]
        emb = self.proj(x)             # [B, 512]
        return emb

    def forward(self, x):
        emb = self.forward_features(x)
        logits = self.classifier(emb)
        return logits, emb
