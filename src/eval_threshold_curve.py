import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from PIL import Image
from scipy.ndimage import gaussian_filter


# --- Config ---
FITS_PATH = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.fits"
CAT_PATH = "data/benchmark/hst_13779_1d_acs_wfc_f606w_jcoi1d_drc.cat"
HEATMAP_PATH = "output_visualizations/heatmap_fixed.png"
HEATMAP_RAW = "output_visualizations/heatmap_raw.npy"  # optional

# thresholds = np.linspace(0.1, 0.25, 10) # e.g., 10 steps from 0.1 to 0.25
thresholds = np.linspace(0.25, 0.8, 12)  # e.g., 12 steps from 0.25 to 0.8
# thresholds = np.linspace(0.01, 0.10, 20) # e.g., 20 steps from 0.01 to 0.10

MATCH_RADIUS = 5.0

# --- Load ground truth ---
sexcat = ascii.read(CAT_PATH)
sex_coords = np.vstack((sexcat["Y_IMAGE"], sexcat["X_IMAGE"])).T

# --- Load model heatmap ---
heatmap = np.load(HEATMAP_RAW)

# Load FITS again to get mask (same logic as in insight.py)
fits_data = fits.open(FITS_PATH)
sci_image = np.nan_to_num(fits_data["SCI"].data)
smoothed = gaussian_filter(sci_image, sigma=5)
valid_mask = smoothed > 1e-3

# Apply mask to heatmap (ignore low signal regions)
heatmap[~valid_mask] = 0.0

print("Heatmap stats:", np.min(heatmap), np.max(heatmap), np.mean(heatmap))


# Alternatively, use the raw array if saved:
# heatmap = np.load(HEATMAP_RAW)

# --- Evaluation loop ---
precision_list = []
recall_list = []
f1_list = []

for thresh in thresholds:
    coords = np.argwhere(heatmap > thresh)

    # Optional: subsample if too large (to simulate num_peaks)
    if coords.shape[0] > 10000:
        coords = coords[np.random.choice(coords.shape[0], 10000, replace=False)]


    print(f"\n--- Threshold: {thresh:.3f} ---")
    print(f"#Coords: {len(coords)}")
    print(f"Pixels above threshold {thresh:.3f}: {(heatmap > thresh).sum()}")

    # SExtractor coords are (x, y), so we flip to (y, x) to match detection layout
    sex_coords_flipped = sex_coords[:, [1, 0]]
    tree = cKDTree(sex_coords_flipped)
    matches = tree.query_ball_point(coords, r=MATCH_RADIUS)

    TP = sum([1 for m in matches if len(m) > 0])
    FP = len(coords) - TP
    FN = len(sex_coords) - TP

    print(f"TP={TP}, FP={FP}, FN={FN}")


    tree = cKDTree(sex_coords)
    matches = tree.query_ball_point(coords, r=MATCH_RADIUS)
    TP = sum([1 for m in matches if len(m) > 0])
    FP = len(coords) - TP
    FN = len(sex_coords) - TP

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision_list, label="Precision")
plt.plot(thresholds, recall_list, label="Recall")
plt.plot(thresholds, f1_list, label="F1 Score")
plt.xlabel("Detection Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, F1 vs. Threshold")
plt.grid()
plt.legend()
uri = "output_visualizations/pr_curve.png"
plt.savefig(uri)
print("Saved: ", uri)
