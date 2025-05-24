from astropy.io import fits
import numpy as np
import torch

def load_fits_tensor(fits_path, normalize=True, clip_percentile=99.5):
    """
    Load a FITS image and convert it to a normalized PyTorch tensor.

    Args:
        fits_path (str): Path to the .fits file
        normalize (bool): Whether to normalize pixel values
        clip_percentile (float): Clip extreme values for robustness

    Returns:
        torch.Tensor: shape (1, H, W), dtype float32, range [0,1]
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data if len(hdul) > 1 else hdul[0].data
        data = np.array(data, dtype=np.float32)

    if normalize:
        finite_data = data[np.isfinite(data)]
        clip_value = np.percentile(finite_data, clip_percentile)
        data = np.clip(data, 0, clip_value) / clip_value

    data = np.nan_to_num(data)
    tensor = torch.from_numpy(data).unsqueeze(0)  # shape: (1, H, W)
    return tensor
