from typing import Any

import numpy as np

from astrapia.callbacks.base import BaseCallback


class ToBCHW(BaseCallback):
    """Normalize and convert image(s) in request.storage to BCHW numpy arrays.

    Expects:
        - `specs.mean` (np.ndarray) and `specs.stnd` (np.ndarray) for per-channel normalization.
        - `request.storage["image"]`: a single HWC/HW array.
        - OR `request.storage["images"]`: a list/tuple of HWC/HW arrays (or None).
    """

    def __init__(self, specs: Any) -> None:
        """
        Args:
            specs: Configuration object with attributes:
                - mean (np.ndarray): per-channel mean to subtract.
                - stnd (np.ndarray): per-channel std to divide.
        """
        self.specs = specs

    def before_process(self, request: Any) -> Any:
        """Hook to run before the main processing step.

        Converts and normalizes:
          - `image` → `tensor` (shape 1xCxHxW or 1x1xHxW)
          - `images` → `tensors` (list of the above or None entries)

        Args:
            request: Object with a `.storage` dict to read from and write to.

        Returns:
            The modified request with added `tensor` or `tensors`.
        """
        # Single image conversion
        if (
            "image" in request.storage
            and isinstance(request.storage["image"], np.ndarray)
            and request.storage["image"].ndim in (2, 3)
        ):
            request.storage["tensor"] = self.to_bchw(request.storage["image"])

        # Batch of images conversion
        if (
            "images" in request.storage
            and isinstance(request.storage["images"], list | tuple)
            and all(isinstance(x, np.ndarray) or x is None for x in request.storage["images"])
            and all(x.ndim in (2, 3) for x in request.storage["images"] if isinstance(x, np.ndarray))
        ):
            request.storage["tensors"] = [x if x is None else self.to_bchw(x) for x in request.storage["images"]]

        return request

    def to_bchw(self, image: np.ndarray) -> np.ndarray:
        """Normalize and reshape an image to BCHW.

        Steps:
          1. Cast to float32.
          2. Subtract `specs.mean` if provided.
          3. Divide by `specs.stnd` if provided.
          4. Add batch & channel dims:
             - Gray (2D) → shape (1,1,H,W)
             - Color (3D HWC) → shape (1,C,H,W)

        Args:
            image (np.ndarray): Input image array.

        Returns:
            np.ndarray: Normalized BCHW tensor.
        """
        image = np.float32(image)
        if hasattr(self.specs, "mean") and isinstance(self.specs.mean, np.ndarray):
            image = image - self.specs.mean
        if hasattr(self.specs, "stnd") and isinstance(self.specs.stnd, np.ndarray):
            image = image / self.specs.stnd

        # HWC to BCHW or HW to BCHW
        return image[None, None] if image.ndim == 2 else np.transpose(image, (2, 0, 1))[None]
