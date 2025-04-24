"""Geometric transformation utilities.

Provides functions to compute and apply similarity transforms
between point sets and images.
"""

__all__ = ["similarity", "source2target_converter"]

import cv2
import numpy as np


def similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute a 2D similarity transformation matrix that maps `source` points to `target` points.

    This implementation is ported from scikit-image (BSD-3-Clause).
    Copyright: 2009-2022 the scikit-image team
    License: BSD-3-Clause (https://scikit-image.org/docs/stable/license.html)
    https://github.com/scikit-image/scikit-image/blob/v0.23.2/skimage/transform/_geometric.py

    Args:
        source (np.ndarray): Array of shape (N, 2) with source coordinates.
        target (np.ndarray): Array of shape (N, 2) with target coordinates.

    Returns:
        np.ndarray: Homogeneous transform matrix of shape (3, 3).

    Notes:
        - Uses procrustes analysis via SVD to estimate scale, rotation, and translation.
        - If the estimated matrix is rank-deficient, returns an identity matrix scaled by NaN.
    """
    source, target = map(np.float64, (source, target))
    dim = source.shape[1]
    s_mu, t_mu = source.mean(axis=0), target.mean(axis=0)
    A = (target - t_mu).T @ (source - s_mu) / source.shape[0]
    valid = np.ones((source.shape[1],), dtype=np.float64)
    if np.linalg.det(A) < 0:
        valid[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, V = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T

    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V

        else:
            save = valid[dim - 1]
            valid[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(valid) @ V
            valid[dim - 1] = save

    else:
        T[:dim, :dim] = U @ np.diag(valid) @ V

    scale = 1.0 / (source - s_mu).var(axis=0).sum() * (S @ valid)
    T[:dim, dim] = t_mu - scale * (T[:dim, :dim] @ s_mu.T)
    T[:dim, :dim] *= scale
    return T


def source2target_converter(
    image: np.ndarray | None,
    points: np.ndarray | None,
    size_hxw: tuple[int, int] | None = None,
    source: np.ndarray | None = None,
    target: np.ndarray | None = None,
    tmat: np.ndarray | None = None,
    invert: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Apply a similarity transform to an image and/or a set of points.

    Args:
        image (np.ndarray | None):
            Input image array (HxWxC). If None, image warping is skipped.
        points (np.ndarray | None):
            Array of shape (N, 2) to be transformed. If None, point projection is skipped.
        size_hxw (tuple[int, int] | None):
            Output size (height, width) for image warping. Required if `image` is provided.
        source (np.ndarray | None):
            Array of source points (N, 2). Used to compute `tmat` if not provided.
        target (np.ndarray | None):
            Array of target points (N, 2). Used to compute `tmat` if not provided.
        tmat (np.ndarray | None):
            Precomputed homogeneous transform matrix of shape (3, 3).
        invert (bool):
            If True, invert the transform to map `target` back to `source`.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]:
            - warped_image (np.ndarray) or None
            - transformed_points (np.ndarray) or None

    Raises:
        ValueError:
            - If neither `tmat` nor both `source` and `target` are valid arrays.
            - If `image` is provided but `size_hxw` is None.
    """
    if not ((isinstance(source, np.ndarray) and isinstance(target, np.ndarray)) or isinstance(tmat, np.ndarray)):
        raise ValueError("source2target_converter: Neither tmat nor (source and target) are valid.")
    if tmat is None:  # only compute when tmat is not available
        n = min(source.shape[0], target.shape[0])
        tmat = similarity(source[:n], target[:n])
    if invert:  # projecting target (image | points) to source
        tmat = np.linalg.inv(tmat)

    warped_image = None
    transformed_points = None
    if image is not None:  # apply affine warp (drop homogeneous row)
        if size_hxw is None:
            raise ValueError("source2target_converter: size_hxw is required to transform image.")
        warped_image = cv2.warpAffine(image, tmat[:2], size_hxw[::-1])

    if points is not None:  # convert to homogeneous coords and apply full matrix
        hom_points = np.concatenate((points, np.ones(points.shape[0])[:, None]), -1)
        proj_points = (tmat @ hom_points.T).T
        transformed_points = proj_points[:, :2] / proj_points[:, [2]]
    return warped_image, transformed_points
