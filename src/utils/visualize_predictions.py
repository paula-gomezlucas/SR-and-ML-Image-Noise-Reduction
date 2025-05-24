import torch
import matplotlib.pyplot as plt
from src.models.hybrid_model import HybridSRDetector
from src.utils.tile_dataset import FITSTileDataset
from torchvision.transforms.functional import normalize

import os

def visualize_predictions(model_path="checkpoints/supervised_model.pt", num_samples=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridSRDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load dataset
    dataset = FITSTileDataset(
    fits_dir="data/benchmark", 
    catalog_dir="data/benchmark"
)

    
    for i in range(num_samples):
        image, gt = dataset[i]
        image_batch = image.unsqueeze(0).to(device)  # shape: [1, 1, 224, 224]
        with torch.no_grad():
            pred = model(image_batch).cpu().squeeze().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image.squeeze().numpy(), cmap="gray")
        axs[0].set_title("Input Tile")
        axs[1].imshow(gt.squeeze().numpy(), cmap="hot")
        axs[1].set_title("Ground Truth Heatmap")
        axs[2].imshow(pred.squeeze(), cmap="hot")
        axs[2].set_title("Predicted Heatmap")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_predictions()
