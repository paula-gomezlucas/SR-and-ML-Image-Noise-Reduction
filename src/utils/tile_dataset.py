import torch
from torch.utils.data import Dataset
import numpy as np
from astropy.io import fits
import os
from .sextractor_loader import load_sextractor_catalog
from typing import List, Tuple

def extract_tiles_and_labels(image: np.ndarray, catalog: np.ndarray, tile_size: int = 224, stride: int = 112, log_file: str = "detections.txt") -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split the large image and associated catalog into smaller tiles and corresponding labels.
    Additionally, log object coordinates to a text file for visualization or analysis.
    """
    tiles = []
    labels = []
    h, w = image.shape

    with open(log_file, 'w') as f:
        f.write("# TileX TileY ObjIdx X_local Y_local\n")

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y:y+tile_size, x:x+tile_size]
                objects = []

                for obj in catalog:
                    x_img, y_img = obj[0], obj[1]
                    if x <= x_img < x + tile_size and y <= y_img < y + tile_size:
                        x_local = x_img - x
                        y_local = y_img - y
                        objects.append((x_local, y_local))

                heatmap = np.zeros((tile_size, tile_size), dtype=np.float32)
                for i, (x_local, y_local) in enumerate(objects):
                    if 0 <= int(y_local) < tile_size and 0 <= int(x_local) < tile_size:
                        heatmap[int(y_local), int(x_local)] = 1.0
                        f.write(f"{x} {y} {i+1} {x_local:.2f} {y_local:.2f}\n")
                    else:
                        print(f"[Warning] Skipped out-of-bounds label: ({x_local}, {y_local})")

                if objects:
                    tiles.append(tile)
                    labels.append(heatmap)

    return tiles, labels

class FITSTileDataset(Dataset):
    def __init__(self, fits_dir, catalog_dir):
        self.fits_files = [f for f in os.listdir(fits_dir) if f.endswith(".fits")]
        if len(self.fits_files) == 0:
            raise ValueError("No FITS files found.")
        self.fits_dir = fits_dir
        self.catalog_dir = catalog_dir
        self.tiles = []
        self.targets = []
        self.tile_size = 224

        for fname in self.fits_files:
            path = os.path.join(fits_dir, fname)
            with fits.open(path) as hdul:
                img_data = hdul[1].data.astype(np.float32)

            # Patch NaNs
            if np.isnan(img_data).any():
                print(f"[Info] Found NaNs in {fname} — replacing with zeros.")
                img_data = np.nan_to_num(img_data, nan=0.0)

            # Load catalog (assumes same base name but .cat extension)
            catalog_name = fname.replace(".fits", ".cat")
            catalog_path = os.path.join(catalog_dir, catalog_name)
            catalog = load_sextractor_catalog(catalog_path)

            tiles, labels = extract_tiles_and_labels(img_data, catalog, tile_size=self.tile_size, log_file="detections.txt")
            self.tiles.extend(tiles)
            self.targets.extend([label[np.newaxis, :, :] for label in labels])  # shape: [1, H, W]

        print(f"Total tiles: {len(self.tiles)}")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = self.tiles[idx]
        target = self.targets[idx]

        # Convert to tensors
        tile_tensor = torch.from_numpy(tile).unsqueeze(0)  # [1, H, W]
        target_tensor = torch.from_numpy(target)

        # Check & sanitize NaNs again
        if torch.isnan(tile_tensor).any():
            print(f"[Warning] NaNs found at idx {idx}, replacing with 0")
            tile_tensor = torch.nan_to_num(tile_tensor, nan=0.0)

        return tile_tensor, target_tensor
