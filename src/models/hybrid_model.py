import torch
import torch.nn as nn
from models.swin_backbone import SwinBackbone
from models.detection_head import HeatmapDetector
from models.stochastic_resonance import apply_stochastic_resonance


class HybridSRDetector(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=1, sr_default_params=None):
        super().__init__()
        self.sr_params = sr_default_params or {
            "noise_level": 0.0,
            "nonlinearity": "identity"
        }

        # Backbone
        self.backbone = SwinBackbone(img_size=img_size, patch_size=patch_size, in_chans=in_chans)

        # Detection head
        num_features = self.backbone.embed_dim  # should match Swin's output dim
        self.head = HeatmapDetector()

    def forward(self, x):
        # Apply stochastic resonance (default SR params used here)
        x = apply_stochastic_resonance(x, self.sr_params)  # shape: [B, C, H, W]

        # Extract features using backbone
        features = self.backbone(x)  # should be [B, C, H, W]
        features = features.permute(0, 3, 1, 2)  # Fix: convert to [B, C, H, W]
        
        if features.dim() == 5:
            # Fix incorrect 5D output
            B, D, H, W, C = features.shape
            features = features[:, -1]  # take last depth
            features = features.permute(0, 3, 1, 2)  # → [B, C, H, W]

        elif features.dim() == 3:
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, C, H, W)

        return self.head(features)  # → Heatmap



