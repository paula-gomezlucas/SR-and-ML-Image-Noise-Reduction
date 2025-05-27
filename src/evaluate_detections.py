import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
from scipy.spatial import cKDTree
import os

# --- Config ---
OUT_DIR = "output_visualizations"
FITS_PATH = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.fits"
CAT_PATH = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.cat"
DETECTIONS_PATH = os.path.join(OUT_DIR, "detections.csv")
MATCH_RADIUS = 5.0

# --- Load data ---
sci_image = fits.getdata(FITS_PATH)
sci_image = np.nan_to_num(sci_image)

model_coords = np.loadtxt(DETECTIONS_PATH, delimiter=",", skiprows=1)
sexcat = ascii.read(CAT_PATH)
sex_coords = np.vstack((sexcat["Y_IMAGE"], sexcat["X_IMAGE"])).T

# --- Match detections ---
tree = cKDTree(sex_coords)
matches = tree.query_ball_point(model_coords, r=MATCH_RADIUS)

TP_idx = [i for i, m in enumerate(matches) if len(m) > 0]
FP_idx = [i for i, m in enumerate(matches) if len(m) == 0]
TP_coords = model_coords[TP_idx]
FP_coords = model_coords[FP_idx]

tree_rev = cKDTree(model_coords)
sex_matches = tree_rev.query_ball_point(sex_coords, r=MATCH_RADIUS)
FN_coords = sex_coords[[i for i, m in enumerate(sex_matches) if len(m) == 0]]

# --- Plot ---
plt.figure(figsize=(10, 10))
plt.imshow(sci_image, cmap='gray', origin='lower', vmin=np.percentile(sci_image, 5), vmax=np.percentile(sci_image, 99))
plt.scatter(FP_coords[:, 1], FP_coords[:, 0], s=10, c="red", label="False Positives")
plt.scatter(TP_coords[:, 1], TP_coords[:, 0], s=10, c="green", label="True Positives")
plt.scatter(FN_coords[:, 1], FN_coords[:, 0], s=10, c="cyan", label="False Negatives")
plt.legend()
plt.title("Annotated Detection Evaluation")
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUT_DIR, "annotated_detections_overlay.png"))
print(f"Saved annotated overlay to {OUT_DIR}")