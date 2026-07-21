
import torch
import torch.nn as nn


class PatchEmbeddingConv(nn.Module):


    def __init__(self, n_channels, emb_dim=40, temporal_kernel=25, pool_size=75, pool_stride=15):
        super().__init__()
        self.temporal_conv = nn.Conv2d(1, emb_dim, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2))
        self.spatial_conv = nn.Conv2d(emb_dim, emb_dim, kernel_size=(n_channels, 1))
        self.bn = nn.BatchNorm2d(emb_dim)
        self.act = nn.ELU()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride))
        self.drop = nn.Dropout(0.3)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.drop(x)

        x = x.squeeze(2).permute(0, 2, 1)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads=8, mlp_ratio=4, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * mlp_ratio, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class EEGConformer(nn.Module):
    def __init__(self, n_channels=16, conv_emb_dim=40, n_layers=4, n_heads=8,
                 out_dim=512, dropout=0.3, n_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbeddingConv(n_channels, emb_dim=conv_emb_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(conv_emb_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(conv_emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(conv_emb_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(out_dim, n_classes)

    def forward_features(self, x):

        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)         
        emb = self.proj(x)        
        return emb

    def forward(self, x):
        emb = self.forward_features(x)
        logits = self.classifier(emb)
        return logits, emb
