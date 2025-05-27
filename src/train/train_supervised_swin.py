import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.tile_dataset import FITSTileDataset
from models.swin_backbone import SwinBackbone
from models.detection_head import HeatmapDetector
from tqdm import tqdm

def train_supervised_swin():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Model ===
    backbone = SwinBackbone(in_chans=1).to(device)
    head = HeatmapDetector().to(device)
    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = optim.Adam(params, lr=1e-4)
    loss_fn = nn.MSELoss()

    # === Data ===
    dataset = FITSTileDataset("data/benchmark", "data/benchmark")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Check FITS/catalog loading.")

    # === Train ===
    num_epochs = 10
    for epoch in range(num_epochs):
        backbone.train()
        head.train()
        total_loss = 0
        nan_batches = 0

        loop = tqdm(dataloader, desc=f"[Epoch {epoch}]")
        for batch_idx, (images, targets) in enumerate(loop):
            images = images.to(device)        # [B, 1, 224, 224]
            targets = targets.to(device)      # [B, 1, 224, 224]

            # === Forward ===
            features = backbone(images)       
            
            # Ensure [B, C, H, W] before passing to head
            if features.dim() == 3:  # [B, N, C]
                B, N, C = features.shape
                H = W = int(N**0.5)
                features = features.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # → [B, C, H, W]
            elif features.dim() == 4 and features.shape[-1] == 768:  # [B, H, W, C]
                features = features.permute(0, 3, 1, 2).contiguous()  # → [B, C, H, W]
            # else: assume it's already in [B, C, H, W]


            heatmap_pred = head(features)     # [B, 1, H, W]
            heatmap_pred = F.interpolate(heatmap_pred, size=(224, 224), mode="bilinear", align_corners=False)

            # === Loss ===
            loss = loss_fn(heatmap_pred, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf loss at batch {batch_idx}, skipping")
                nan_batches += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            avg_loss_so_far = total_loss / (batch_idx + 1)
            loop.set_postfix(loss=avg_loss_so_far)

        avg_loss = total_loss / (len(dataloader) - nan_batches) if (len(dataloader) - nan_batches) > 0 else float("nan")
        print(f"[Epoch {epoch}] Avg Loss = {avg_loss:.6f}, Skipped = {nan_batches}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "backbone": backbone.state_dict(),
        "head": head.state_dict()
    }, "checkpoints/swin_supervised.pt")
    print("[Saved] checkpoints/swin_supervised.pt")

if __name__ == "__main__":
    train_supervised_swin()
