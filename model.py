"""
model.py
--------
CNN + FNN Fusion model for cloudburst detection.

Architecture:
    CNN Branch  : Conv2D × 3 → GlobalAvgPool → Dense(256)
    FNN Branch  : Dense 4→64→128→64
    Fusion Head : Concat(256+64=320) → Dense(128) → Dense(64) → Dense(1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  CNN Branch  (processes satellite images)
# ──────────────────────────────────────────────────────────────
class CNNBranch(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 224 → 112
            nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 112 → 56
            nn.Dropout2d(0.2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 56 → 28
            nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 28 → 14
        )

        # Global Average Pooling  → (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.head(x)
        return x   # (B, 256)


# ──────────────────────────────────────────────────────────────
#  FNN Branch  (processes ERA5 tabular features)
# ──────────────────────────────────────────────────────────────
class FNNBranch(nn.Module):
    def __init__(self, in_features=4, out_features=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(128, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)   # (B, 64)


# ──────────────────────────────────────────────────────────────
#  Fusion Head  (combines CNN + FNN features)
# ──────────────────────────────────────────────────────────────
class CloudburstFusionModel(nn.Module):
    """
    Full multimodal model:
        image  → CNNBranch  → (B, 256)  ──┐
                                            ├─ Concat → (B, 320) → classifier → (B, 1)
        tabular → FNNBranch → (B, 64)  ──┘
    """

    def __init__(self, cnn_out=256, fnn_out=64):
        super().__init__()

        self.cnn = CNNBranch(out_features=cnn_out)
        self.fnn = FNNBranch(in_features=4, out_features=fnn_out)

        fusion_in = cnn_out + fnn_out   # 320

        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(64, 1),           # raw logit (no sigmoid — BCEWithLogitsLoss)
        )

    def forward(self, image, tabular):
        cnn_feat = self.cnn(image)       # (B, 256)
        fnn_feat = self.fnn(tabular)     # (B, 64)
        fused    = torch.cat([cnn_feat, fnn_feat], dim=1)   # (B, 320)
        logit    = self.classifier(fused)                   # (B, 1)
        return logit.squeeze(1)          # (B,)


# ──────────────────────────────────────────────────────────────
#  Quick architecture summary
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CloudburstFusionModel()
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # Dummy forward pass
    dummy_img = torch.randn(4, 3, 224, 224)
    dummy_tab = torch.randn(4, 4)
    out = model(dummy_img, dummy_tab)
    print(f"Output shape: {out.shape}  (expected: [4])")
