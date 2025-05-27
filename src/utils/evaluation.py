import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

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


def match_detections(gt_coords, pred_coords, threshold=6.0):
    """
    Match predicted detections to GT points using greedy distance matching.

    Args:
        gt_coords (np.ndarray): Ground truth coordinates [N_gt, 2]
        pred_coords (np.ndarray): Predicted coordinates [N_pred, 2]
        threshold (float): Maximum distance for a TP match

    Returns:
        tp_indices (List[int]): Indices of pred_coords that are TPs
        fp_indices (List[int]): Indices of pred_coords that are FPs
        fn_indices (List[int]): Indices of gt_coords that are FNs
    """

    if len(gt_coords) == 0 and len(pred_coords) == 0:
        return [], [], []

    print("pred_coords:", pred_coords)
    print("gt_coords:", gt_coords)
    print("pred_coords shape:", np.shape(pred_coords))
    print("gt_coords shape:", np.shape(gt_coords))


    dist_matrix = cdist(pred_coords, gt_coords)
    matched_gt = set()
    matched_pred = set()

    for pred_idx, row in enumerate(dist_matrix):
        gt_idx = np.argmin(row)
        if row[gt_idx] <= threshold and gt_idx not in matched_gt:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

    tp = list(matched_pred)
    fp = [i for i in range(len(pred_coords)) if i not in matched_pred]
    fn = [i for i in range(len(gt_coords)) if i not in matched_gt]

    return tp, fp, fn


def evaluate_tile(ground_truth, detections, radius=6.0):
    TP, FP, FN = match_detections(ground_truth, detections, radius)
    print(f"TP: {len(TP)} | FP: {len(FP)} | FN: {len(FN)}")
    return TP, FP, FN


# Aliases for compatibility with existing code
compute_precision_recall = compute_reward 
# match_detections_to_ground_truth = match_detections_to_gt