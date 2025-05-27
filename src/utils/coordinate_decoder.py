import numpy as np
from scipy.ndimage import maximum_filter, center_of_mass

def decode_coordinates_from_heatmap(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    refine_method: str = "centroid",
    tile_origin: tuple = (0, 0),
    return_global: bool = False
):
    """
    Decode coordinates from a heatmap, optionally shifting to global coordinates.

    Returns:
        dict: {
            "tile_origin": (tile_x, tile_y),
            "local_coords": [(x, y), ...],
            "global_coords": [(x, y), ...] if return_global is True
        }
    """
    if heatmap.ndim != 2:
        raise ValueError(f"Expected 2D heatmap, got shape {heatmap.shape}")

    tile_x, tile_y = tile_origin
    mask = heatmap > threshold
    if not np.any(mask):
        return {
            "tile_origin": (tile_x, tile_y),
            "local_coords": [],
            "global_coords": [] if return_global else None
        }

    peaks = (maximum_filter(heatmap, size=3) == heatmap) & mask
    peak_indices = np.argwhere(peaks)

    local_coords = []
    for y, x in peak_indices:
        if refine_method == "centroid":
            window = heatmap[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            local_mask = window >= threshold
            if np.any(local_mask):
                cy, cx = center_of_mass(window * local_mask)
                refined_x = x - 1 + cx
                refined_y = y - 1 + cy
                local_coords.append((refined_x, refined_y))
            else:
                local_coords.append((x, y))
        else:
            local_coords.append((x, y))

    global_coords = [(x + tile_x, y + tile_y) for x, y in local_coords] if return_global else None

    return {
        "tile_origin": (tile_x, tile_y),
        "local_coords": local_coords,
        "global_coords": global_coords
    }
