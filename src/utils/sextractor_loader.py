import pandas as pd
import numpy as np
from io import StringIO

def load_cat_as_array(cat_path):
    """
    Load a SExtractor .cat file into an Nx2 or Nx3 numpy array.

    Assumes columns: [ID, X_IMAGE, Y_IMAGE, ...]

    Args:
        cat_path (str): Path to the .cat file

    Returns:
        np.ndarray: Nx2 or Nx3 array with object positions
    """
    with open(cat_path, 'r') as f:
        lines = [line for line in f if not line.startswith('#') and line.strip()]

    df = pd.read_csv(StringIO(''.join(lines)), sep=r'\s+', header=None)
    if df.shape[1] < 3:
        raise ValueError("Expected at least 3 columns (ID, X, Y)")

    return df[[1, 2]].to_numpy(dtype=np.float32)

load_sextractor_catalog = load_cat_as_array  # alias for compatibility