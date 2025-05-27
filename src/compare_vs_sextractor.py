from astropy.io import ascii
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os

# Load SExtractor detections
sexcat_path = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.cat"
sexcat = ascii.read(sexcat_path)
sex_coords = np.vstack((sexcat["Y_IMAGE"], sexcat["X_IMAGE"])).T  # Shape: (N, 2)
OUT_DIR = "output_visualizations"

# Load your model detections
model_coords = np.loadtxt(os.path.join(OUT_DIR, "detections.csv"), delimiter=",", skiprows=1)

# Match detections within radius
from scipy.spatial import cKDTree

tree = cKDTree(sex_coords)
matches = tree.query_ball_point(model_coords, r=5.0)  # radius in pixels

TP = sum([1 for match in matches if len(match) > 0])        # True Positives
FP = len(model_coords) - TP                                 # False Positives
FN = len(sex_coords) - TP                                   # False Negatives

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"TP={TP}, FP={FP}, FN={FN}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# Load FITS image for background
fits_path = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.fits"
sci_image = fits.getdata(fits_path)
sci_image = np.nan_to_num(sci_image)

plt.figure(figsize=(10, 10))
plt.imshow(sci_image, cmap='gray', origin='lower', vmin=np.percentile(sci_image, 5), vmax=np.percentile(sci_image, 99))
plt.scatter(model_coords[:, 1], model_coords[:, 0], s=10, c="red", label="Model")
plt.scatter(sex_coords[:, 1], sex_coords[:, 0], s=10, c="cyan", label="SExtractor", alpha=0.5)
plt.legend()
plt.title("Model vs. SExtractor Detections")
os.makedirs("output_visualizations", exist_ok=True)
plt.savefig(os.path.join("output_visualizations", "comparison_overlay.png"))