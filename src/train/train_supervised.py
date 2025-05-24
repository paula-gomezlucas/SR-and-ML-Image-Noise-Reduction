import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.hybrid_model import HybridSRDetector
from utils.tile_dataset import FITSTileDataset
from utils.evaluation import compute_precision_recall
from utils.sextractor_loader import load_sextractor_catalog
from tqdm import tqdm


def check_tensor_nan_inf(name, tensor):
    if torch.isnan(tensor).any():
        print(f"[Warning] {name} contains NaNs")
    if torch.isinf(tensor).any():
        print(f"[Warning] {name} contains Infs")


def train_supervised():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridSRDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    fits_dir = "data/benchmark"
    cat_dir = "data/benchmark"

    dataset = FITSTileDataset(fits_dir, cat_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Check FITS/catalog loading.")

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        nan_batches = 0
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            images = images.to(device)
            targets = targets.to(device)

            # Check inputs
            check_tensor_nan_inf("images", images)
            check_tensor_nan_inf("targets", targets)

            preds = model(images)

            # Check outputs
            check_tensor_nan_inf("preds (before upsampling)", preds)

            preds_upsampled = F.interpolate(preds, size=(224, 224), mode='bilinear', align_corners=False)

            # Check upsampled output
            check_tensor_nan_inf("preds_upsampled", preds_upsampled)

            loss = loss_fn(preds_upsampled, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN or Inf loss at batch {batch_idx}, skipping...")
                nan_batches += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if len(dataloader) - nan_batches == 0:
            print("[Error] All batches produced NaN loss.")
            avg_loss = float('nan')
        else:
            avg_loss = total_loss / (len(dataloader) - nan_batches)
        print(f"Epoch {epoch}: Avg Loss={avg_loss:.4f} (Skipped {nan_batches} batches due to NaNs)")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/supervised_model.pt")
    print("[Info] Model saved to checkpoints/supervised_model.pt")

if __name__ == "__main__":
    train_supervised()
