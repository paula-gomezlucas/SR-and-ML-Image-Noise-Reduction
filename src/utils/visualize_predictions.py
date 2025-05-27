import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.hybrid_model import HybridSRDetector
from utils.tile_dataset import FITSTileDataset
from utils.coordinate_decoder import decode_coordinates_from_heatmap
from torchvision.transforms.functional import normalize

import os

def visualize_predictions(model_path="../../checkpoints/supervised_model.pt", num_samples=5, use_global_coords=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridSRDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load dataset
    dataset = FITSTileDataset(
        fits_dir="../../data/benchmark", 
        catalog_dir="../../data/benchmark"
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    max_visuals_to_show = 2
    visual_count = 0

    for batch_idx, (images, gts, origins) in enumerate(dataloader):
        images = images.to(device)
        with torch.no_grad():
            preds = model(images).cpu().numpy()  # shape: [B, 1, 224, 224]

        for i in range(images.shape[0]):
            if visual_count >= num_samples:
                return

            pred = preds[i, 0]
            image = images[i, 0].cpu().numpy()
            gt = gts[i, 0].cpu().numpy()
            origin_tensor = origins[i]  # Tensor of shape [2]
            origin = tuple(origin_tensor.tolist())  # Ensure (tile_x, tile_y)

            if not isinstance(origin, tuple) or len(origin) != 2:
                raise ValueError(f"Expected origin to be 2 elements (tile_x, tile_y), got {origin}")

            result = decode_coordinates_from_heatmap(
                pred,
                threshold=0.005,
                refine_method="centroid",
                tile_origin=origin,
                return_global=True
            )

            coords = result["global_coords"] if use_global_coords else result["local_coords"]

            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(image, cmap="gray")
            axs[0].set_title("Input Tile")
            axs[1].imshow(gt, cmap="hot")
            axs[1].set_title("Ground Truth Heatmap")
            axs[2].imshow(pred, cmap="hot")
            axs[2].set_title("Predicted Heatmap")
            for ax in axs:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

            visual_count += 1

if __name__ == "__main__":
    print("=== GLOBAL COORDINATES ===")
    visualize_predictions(use_global_coords=True)

    print("=== LOCAL COORDINATES ===")
    visualize_predictions(use_global_coords=False)
