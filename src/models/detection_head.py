import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapDetector(nn.Module):
    def __init__(self, threshold=0.3, min_distance=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),  # match Swin output channels
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.threshold = threshold
        self.min_distance = min_distance

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [B, C=768, H, W] from Swin

        Returns:
            torch.Tensor: [B, 1, H, W] heatmap output
        """
        return self.encoder(x)  # Output heatmap for each tile
