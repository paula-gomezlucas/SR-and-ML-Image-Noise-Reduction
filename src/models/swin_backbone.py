import torch.nn as nn
from timm.models.swin_transformer import swin_tiny_patch4_window7_224


class SwinBackbone(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=1):
        super().__init__()
        # Load a pre-trained Swin model (timm)
        self.backbone = swin_tiny_patch4_window7_224(pretrained=True, features_only=True, in_chans=in_chans)
        self.embed_dim = self.backbone.feature_info[-1]['num_chs']

    def forward(self, x):
        feats = self.backbone(x)  # List[Tensor]
        x = feats[-1]  # Last stage

        if x.dim() == 3:  # [B, N, C]
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.transpose(1, 2).reshape(B, C, H, W)

        elif x.dim() == 5:  # [B, D, H, W, C]
            B, D, H, W, C = x.shape
            x = x[:, -1]  # take last depth slice → [B, H, W, C]
            x = x.permute(0, 3, 1, 2)  # → [B, C, H, W]

        return x




