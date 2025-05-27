import pandas as pd
import numpy as np
from io import StringIO

def load_sextractor_catalog(path, x_col='X_IMAGE', y_col='Y_IMAGE'):
    """
    Loads the ground truth coordinates from a SExtractor .cat file (ASCII_HEAD format).
    
    Args:
        path (str): Path to the .cat file.
        x_col (str): Name of the x-coordinate column (default: 'X_IMAGE').
        y_col (str): Name of the y-coordinate column (default: 'Y_IMAGE').

    Returns:
        np.ndarray: Array of shape [N, 2] with (x, y) positions.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find column names (starts with #)
    header_lines = [l for l in lines if l.startswith('#')]
    col_map = {}
    for line in header_lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == '#':
            col_index = int(parts[1]) - 1
            col_name = parts[2]
            col_map[col_name] = col_index

    if x_col not in col_map or y_col not in col_map:
        raise ValueError(f"Columns {x_col} or {y_col} not found in catalog header.")

    x_idx = col_map[x_col]
    y_idx = col_map[y_col]

    # Parse data lines
    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
    coords = []
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) > max(x_idx, y_idx):
            x = float(parts[x_idx])
            y = float(parts[y_idx])
            coords.append((x, y))

    return np.array(coords)


# Aliases
load_cat_as_array = load_sextractor_catalog # alias for compatibility
