import os
import time
import numpy as np
import torch
from tqdm import tqdm
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from models.swin_backbone import SwinBackbone
from models.detection_head import HeatmapDetector
from utils.coordinate_decoder import decode_coordinates_from_heatmap

TILE_SIZE = 224
STRIDE = 112
FITS_PATH = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.fits"
MODEL_PATH = "checkpoints/swin_supervised.pt"
OUT_DIR = "output_visualizations"
tqdm_args = {"file": sys.stdout, "dynamic_ncols": True, "leave": True}

# Load FITS image and clean NaNs
hdul = fits.open(FITS_PATH)
sci_image = np.nan_to_num(hdul["SCI"].data)

if not sci_image.dtype.isnative:
    sci_image = sci_image.byteswap().view(sci_image.dtype.newbyteorder('='))

H, W = sci_image.shape

# Generate validity mask: ignore low-signal regions (e.g., CCD gap or padded borders)
smoothed = gaussian_filter(sci_image, sigma=5)
valid_mask = smoothed > 1e-3

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load trained Swin + Head model
backbone = SwinBackbone(in_chans=1).to(device)
head = HeatmapDetector().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
backbone.load_state_dict(checkpoint["backbone"])
head.load_state_dict(checkpoint["head"])
backbone.eval()
head.eval()

# Create heatmap canvas
heatmap = np.zeros((H, W))
count_map = np.zeros((H, W))

# Tiling loop
y_coords = range(0, H - TILE_SIZE + 1, STRIDE)
x_coords = range(0, W - TILE_SIZE + 1, STRIDE)
total_tiles = len(y_coords) * len(x_coords)

print("Total tiles:", total_tiles)
tile_idx = 0

with tqdm(total=total_tiles, desc="Processing Tiles", **tqdm_args) as pbar:
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            tile = sci_image[y:y + TILE_SIZE, x:x + TILE_SIZE]
            tile_tensor = torch.tensor(tile, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                features = backbone(tile_tensor)
                if features.dim() == 3:
                    B, N, C = features.shape
                    H_ = W_ = int(N ** 0.5)
                    features = features.view(B, H_, W_, C).permute(0, 3, 1, 2).contiguous()
                elif features.dim() == 4 and features.shape[-1] == 768:
                    features = features.permute(0, 3, 1, 2).contiguous()
                heatmap_pred = head(features)
                heatmap_pred = torch.nn.functional.interpolate(
                    heatmap_pred, size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False
                )
                # Optional: normalize each tile's prediction for sharper peaks
                heatmap_pred -= heatmap_pred.min()
                heatmap_pred /= (heatmap_pred.max() + 1e-6)

            heatmap[y:y + TILE_SIZE, x:x + TILE_SIZE] += heatmap_pred.squeeze().cpu().numpy()
            count_map[y:y + TILE_SIZE, x:x + TILE_SIZE] += 1

            if tile_idx % 250 == 0:
                print(f"[Tile {tile_idx}] pred mean: {heatmap_pred.mean().item():.4f}, max: {heatmap_pred.max().item():.4f}")
            tile_idx += 1

            if tile_idx in [0, 500, 1000]:
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(tile_tensor.squeeze().cpu().numpy(), cmap='gray')
                axs[0].set_title("Tile Input")
                axs[1].imshow(heatmap_pred.squeeze().cpu().numpy(), cmap='inferno')
                axs[1].set_title("Tile Prediction")
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f"tile_{tile_idx}_debug.png"))
                plt.close()

            pbar.update(1)

# Normalize heatmap
normalized_heatmap = heatmap / np.maximum(count_map, 1)
normalized_heatmap = np.clip(normalized_heatmap, 0, None)

# Apply the valid mask (CCD gap and empty corners)
normalized_heatmap[~valid_mask] = 0.0

print("Heatmap stats:", normalized_heatmap.min(), normalized_heatmap.max(), normalized_heatmap.mean())

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Save raw heatmap
np.save(os.path.join(OUT_DIR, "heatmap_raw.npy"), normalized_heatmap)

# Save result visualizations
plt.figure(figsize=(10, 10))
plt.imshow(normalized_heatmap, cmap="inferno", origin="lower")
plt.title("Mapa de calor de predicciones del modelo")
plt.colorbar()
plt.savefig(os.path.join(OUT_DIR, "heatmap_fixed.png"))

# Extract detections
threshold = 0.0015
print("Heatmap max value:", normalized_heatmap.max())
start = time.time()
coords = decode_coordinates_from_heatmap(
    normalized_heatmap,
    threshold=threshold,
    min_distance=4,
    max_peaks=np.inf,
    refine_method="gaussian"
)
print(f"Detection took {time.time() - start:.2f} seconds")

# Save histogram
plt.figure()
plt.hist(normalized_heatmap.ravel(), bins=100)
plt.title("Heatmap Value Distribution")
plt.savefig(os.path.join(OUT_DIR, "heatmap_hist.png"))

# Save detections to CSV
np.savetxt(os.path.join(OUT_DIR, "detections.csv"), coords, fmt='%d', delimiter=",", header="y,x", comments="")

print(f"Detected {len(coords)} objects (threshold={threshold})")

# Overlay detections
plt.figure(figsize=(10, 10))
plt.imshow(sci_image, cmap="gray", origin="lower")
plt.scatter(coords[:, 1], coords[:, 0], s=10, c="red", marker="x", label="Detected objects")
plt.title("Detected Objects Overlay")
plt.legend()
plt.savefig(os.path.join(OUT_DIR, "overlay_with_detections.png"))
