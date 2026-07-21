"""
Cross-modal fusion transformer.

Consumes the three FROZEN 512-D embeddings (EEG, gaze, gait) and learns
cross-modal attention between them instead of naive concatenation, so the
model can down-weight a noisy/missing modality per sample.

Supports missing modalities at inference time (since EEG, gaze and gait were
never recorded on the same subjects) via a learned mask token that replaces
any modality embedding that is unavailable for a given sample.
"""

import torch
import torch.nn as nn


class CrossModalFusionTransformer(nn.Module):
    def __init__(self, emb_dim=512, n_heads=8, n_layers=2, mlp_ratio=2,
                 dropout=0.3, n_classes=2):
        super().__init__()
        self.emb_dim = emb_dim

        # learned per-modality type embeddings + a mask token for missing modalities
        self.modality_type_emb = nn.Parameter(torch.randn(3, emb_dim) * 0.02)
        self.missing_token = nn.Parameter(torch.randn(1, emb_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads, dim_feedforward=emb_dim * mlp_ratio,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(emb_dim)

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, eeg_emb=None, gaze_emb=None, gait_emb=None, return_attention=False):
        """
        Each *_emb is either [B, emb_dim] or None (whole modality missing for
        this batch/sample set - replaced by the learned missing_token).
        """
        batch_size = next(e.shape[0] for e in (eeg_emb, gaze_emb, gait_emb) if e is not None)
        device = next(e.device for e in (eeg_emb, gaze_emb, gait_emb) if e is not None)

        tokens = []
        for i, emb in enumerate((eeg_emb, gaze_emb, gait_emb)):
            if emb is None:
                tok = self.missing_token.expand(batch_size, -1)
            else:
                tok = emb
            tok = tok + self.modality_type_emb[i]
            tokens.append(tok.unsqueeze(1))            # [B, 1, emb_dim]

        seq = torch.cat(tokens, dim=1)                  # [B, 3, emb_dim]
        cls = self.cls_token.expand(batch_size, -1, -1).to(device)
        seq = torch.cat([cls, seq], dim=1)               # [B, 4, emb_dim]

        fused = self.transformer(seq)
        fused = self.norm(fused)
        shared = fused[:, 0]                             # CLS token = fused representation

        logits = self.classifier(shared)

        if return_attention:
            return logits, shared, fused
        return logits, shared
