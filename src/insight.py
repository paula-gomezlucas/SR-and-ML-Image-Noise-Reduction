import os
import numpy as np
import torch
from astropy.io import fits
import matplotlib.pyplot as plt
from models.hybrid_model import HybridSRDetector  # Adjust if your class is named differently

TILE_SIZE = 224
STRIDE = 112
FITS_PATH = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.fits"
MODEL_PATH = "checkpoints/supervised_model.pt"
OUT_DIR = "output_visualizations"

# Load FITS image and clean NaNs
hdul = fits.open(FITS_PATH)
sci_image = np.nan_to_num(hdul["SCI"].data)
H, W = sci_image.shape

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridSRDetector().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Create heatmap canvas
heatmap = np.zeros((H, W))
count_map = np.zeros((H, W))

# Predict over each tile and place prediction in heatmap
for y in range(0, H - TILE_SIZE + 1, STRIDE):
    for x in range(0, W - TILE_SIZE + 1, STRIDE):
        tile = sci_image[y:y + TILE_SIZE, x:x + TILE_SIZE]
        tile_native = tile.byteswap().view(tile.dtype.newbyteorder('='))
        tile_tensor = torch.tensor(tile_native, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tile_tensor)
            pred = output.mean().item()  # scalar summary of tile prediction
        
        heatmap[y:y + TILE_SIZE, x:x + TILE_SIZE] += pred
        count_map[y:y + TILE_SIZE, x:x + TILE_SIZE] += 1

# Normalize accumulated predictions
normalized_heatmap = heatmap / np.maximum(count_map, 1)

# Save result
os.makedirs(OUT_DIR, exist_ok=True)
plt.figure(figsize=(10, 10))
plt.imshow(normalized_heatmap, cmap="inferno", origin="lower")
plt.title("Mapa de calor de predicciones del modelo")
plt.colorbar()
plt.savefig(os.path.join(OUT_DIR, "heatmap_fixed.png"))
