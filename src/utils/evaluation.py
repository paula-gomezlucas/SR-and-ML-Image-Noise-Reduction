import numpy as np
from scipy.spatial import cKDTree

def compute_reward(detected, ground_truth, radius=5):
    """
    Compute reward based on proximity between detected and ground truth object positions.

    Args:
        detected (np.ndarray): Nx2 array of [x, y] detections.
        ground_truth (np.ndarray): Mx2 array of [x, y] true object positions.
        radius (float): Matching radius in pixels.

    Returns:
        float: reward = TP / (TP + FP + FN) (F1-inspired)
    """
    if len(detected) == 0 and len(ground_truth) == 0:
        return 1.0
    if len(detected) == 0 or len(ground_truth) == 0:
        return 0.0

    tree_gt = cKDTree(ground_truth)
    tree_det = cKDTree(detected)

    gt_to_det = tree_gt.query_ball_tree(tree_det, r=radius)
    det_to_gt = tree_det.query_ball_tree(tree_gt, r=radius)

    matched_gt = sum(len(matches) > 0 for matches in gt_to_det)
    matched_det = sum(len(matches) > 0 for matches in det_to_gt)

    TP = matched_gt
    FP = len(detected) - matched_det
    FN = len(ground_truth) - matched_gt

    denom = TP + FP + FN
    if denom == 0:
        return 0.0

    reward = TP / denom  # Balanced score
    return reward

compute_precision_recall = compute_reward  # alias for compatibility