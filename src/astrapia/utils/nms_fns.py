__all__ = ["nms_numba"]

import numba
import numpy as np


@numba.jit(nopython=True)
def nms_numba(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Non-maximum suppression with numba (faster when #boxes <= 8096, after which it matches numpy version).

    Args:
        boxes: np.ndarray
            bounding boxes in cornerform
        scores: np.ndarray
            prediction confidence
        threshold: float
            minimum iou threshold required to retain boxes
    """
    n, _ = boxes.shape

    # compute area
    areas = np.zeros(n, dtype=np.float32)
    for i in numba.prange(n):  # pylint: disable=not-an-iterable
        areas[i] = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)

    order = scores.argsort()[::-1]
    keep = np.ones(n, dtype=np.int32)
    for i in range(n):
        if not keep[order[i]]:
            continue

        for j in numba.prange(i + 1, n):  # pylint: disable=not-an-iterable
            if not keep[order[j]]:
                continue

            w = max(0.0, min(boxes[order[i], 2], boxes[order[j], 2]) - max(boxes[order[i], 0], boxes[order[j], 0]) + 1)
            h = max(0.0, min(boxes[order[i], 3], boxes[order[j], 3]) - max(boxes[order[i], 1], boxes[order[j], 1]) + 1)
            intersection = w * h
            iou = intersection / (areas[order[i]] + areas[order[j]] - intersection)
            if iou > threshold:
                keep[order[j]] = 0

    return keep
