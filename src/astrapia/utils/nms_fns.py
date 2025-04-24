__all__ = ["nms_numba"]

import numba
import numpy as np


@numba.jit(nopython=True)
def nms_numba(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Perform non-maximum suppression (NMS) on a set of bounding boxes.

    This Numba-accelerated implementation is faster than a pure NumPy version
    for up to ~8,096 boxes; beyond that, performance is comparable.

    Args:
        boxes (np.ndarray): Array of shape (N, 4) in corner form [x1, y1, x2, y2].
        scores (np.ndarray): Array of length N containing confidence scores.
        threshold (float): IoU threshold above which boxes are suppressed.

    Returns:
        np.ndarray: Integer mask of length N where 1 indicates the box is kept
                    and 0 indicates it is suppressed.
    """
    n, _ = boxes.shape

    # compute area for each box
    areas = np.zeros(n, dtype=np.float32)
    for i in numba.prange(n):  # pylint: disable=not-an-iterable
        areas[i] = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
    # sort boxes by descending score
    order = scores.argsort()[::-1]
    keep = np.ones(n, dtype=np.int32)
    # iterate through boxes, suppressing overlaps
    for i in range(n):
        if not keep[order[i]]:
            continue

        for j in numba.prange(i + 1, n):  # pylint: disable=not-an-iterable
            if not keep[order[j]]:
                continue

            # compute intersection
            w = max(0.0, min(boxes[order[i], 2], boxes[order[j], 2]) - max(boxes[order[i], 0], boxes[order[j], 0]) + 1)
            h = max(0.0, min(boxes[order[i], 3], boxes[order[j], 3]) - max(boxes[order[i], 1], boxes[order[j], 1]) + 1)
            intersection = w * h
            # compute IoU and suppress if above threshold
            iou = intersection / (areas[order[i]] + areas[order[j]] - intersection)
            if iou > threshold:
                keep[order[j]] = 0

    return keep
