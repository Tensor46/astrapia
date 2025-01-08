__all__ = ["similarity", "source2target_converter"]

import cv2
import numpy as np


def similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute similarty transform to convert source to target."""
    # This function is ported from Scikit-Image.
    # Copyright: 2009-2022 the scikit-image team
    # License: BSD-3-Clause (https://scikit-image.org/docs/stable/license.html)
    # https://github.com/scikit-image/scikit-image/blob/v0.23.2/skimage/transform/_geometric.py
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
    if not ((isinstance(source, np.ndarray) and isinstance(target, np.ndarray)) or isinstance(tmat, np.ndarray)):
        raise ValueError("source2target_converter: Neither tmat nor (source and target) are valid.")
    if tmat is None:  # only compute when tmat is not available
        n = min(source.shape[0], target.shape[0])
        tmat = similarity(source[:n], target[:n])
    if invert:  # projecting target (image | points) to source
        tmat = np.linalg.inv(tmat)

    if image is not None:  # warp image
        if size_hxw is None:
            raise ValueError("source2target_converter: size_hxw is required to transform image.")
        image = cv2.warpAffine(image, tmat[:2], size_hxw[::-1])

    if points is not None:  # project points
        points = np.concatenate((points, np.ones(points.shape[0])[:, None]), -1)
        points = (tmat @ points.T).T
        points = points[:, :2] / points[:, [2]]
    return image, points
