import torch
import torch.nn as nn
import numpy as np

from models.swin_backbone import SwinBackbone
from models.detection_head import HeatmapDetector
from models.stochastic_resonance import apply_stochastic_resonance
from models.rl_policy import RLPolicy, sample_action


class HybridSRRLModel(nn.Module):
    """
    A hybrid object detection model with RL-controlled stochastic resonance preprocessing
    and a Swin Transformer backbone with heatmap output.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=1, sr_policy_channels=1, sr_policy_path=None):
        super().__init__()

        # RL Policy for adaptive SR
        self.policy = RLPolicy(input_channels=sr_policy_channels)
        if sr_policy_path is not None:
            self.policy.load_state_dict(torch.load(sr_policy_path, map_location="cpu", weights_only=True))
        self.policy.eval()

        # Swin backbone
        self.backbone = SwinBackbone(img_size=img_size, patch_size=patch_size, in_chans=in_chans)
        self.head = HeatmapDetector()

    def forward(self, x, return_sr_action=False):
        """
        Forward pass through RLPolicy → SR → Swin + detection head, batched.

        Args:
            x (torch.Tensor): [B, 1, H, W] batch of tiles

        Returns:
            torch.Tensor: [B, 1, H, W] heatmaps
        """
        if self.training:
            raise RuntimeError("HybridSRRLModel is for inference. RLPolicy is not trained jointly.")

        B = x.size(0)
        device = x.device

        with torch.no_grad():
            # Get SR actions for the whole batch
            noise_level, nl_logits = self.policy(x)  # [B, 1], [B, 3]
            nl_indices = nl_logits.argmax(dim=1)     # [B]
            nonlinearities = ['identity', 'relu', 'tanh']
            nonlinearity_list = [nonlinearities[i.item()] for i in nl_indices]

            # Apply SR to each tile using vectorized logic
            sr_tiles = []
            for i in range(B):
                action = {
                    'noise_level': noise_level[i].item(),
                    'nonlinearity': nonlinearity_list[i]
                }
                sr_tile = apply_stochastic_resonance(x[i, 0].cpu().numpy(), action)
                if isinstance(sr_tile, np.ndarray):
                    sr_tile = torch.from_numpy(sr_tile)
                sr_tiles.append(sr_tile.unsqueeze(0))


            sr_batch = torch.stack(sr_tiles).to(device)  # [B, 1, H, W]

            # Feature extraction
            features = self.backbone(sr_batch)  # Swin output (varies in shape)

            if features.dim() == 5:
                B_, D, H, W, C = features.shape
                features = features[:, -1].permute(0, 3, 1, 2)  # [B, C, H, W]
            elif features.dim() == 3:
                B_, N, C = features.shape
                H = W = int(N ** 0.5)
                features = features.transpose(1, 2).reshape(B_, C, H, W)
            elif features.dim() == 4 and features.shape[1] < features.shape[-1]:
                features = features.permute(0, 3, 1, 2)  # [B, C, H, W]

            # Detection head
            heatmaps = self.head(features)  # [B, 1, H, W]

        if return_sr_action:
            return heatmaps, {'noise_level': noise_level, 'nonlinearity': nl_indices}

        return heatmaps