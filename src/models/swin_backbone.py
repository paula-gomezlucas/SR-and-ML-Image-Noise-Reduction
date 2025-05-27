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
        return feats[-1]  # [B, C, H, W]





