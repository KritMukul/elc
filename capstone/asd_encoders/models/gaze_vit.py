"""
Vision Transformer encoder for gaze heatmap/scanpath images.
Wraps a timm ViT backbone (pretrained on ImageNet for transfer learning,
critical given only ~25 gaze participants) and projects to a 512-D embedding.
"""

import torch
import torch.nn as nn
import timm


class GazeViT(nn.Module):
    def __init__(self, backbone="vit_small_patch16_224", pretrained=True,
                 out_dim=512, dropout=0.3, n_classes=2, freeze_blocks=6):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        # Freeze the earliest transformer blocks - low-level ViT features
        # transfer fine from ImageNet; only fine-tune the later blocks given
        # the tiny gaze dataset (~25 subjects).
        if hasattr(self.backbone, "blocks") and freeze_blocks > 0:
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
            for i, blk in enumerate(self.backbone.blocks):
                if i < freeze_blocks:
                    for p in blk.parameters():
                        p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward_features(self, x):
        feat = self.backbone(x)      # [B, feat_dim]
        emb = self.proj(feat)        # [B, 512]
        return emb

    def forward(self, x):
        emb = self.forward_features(x)
        logits = self.classifier(emb)
        return logits, emb
