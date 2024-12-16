from typing import Any

import numpy as np

from astrapia.callbacks.base import BaseCallback


class ToBCHW(BaseCallback):
    """Requires mean & stnd in specs, and expects 'image' | 'images' in request.storage."""

    def __init__(self, specs: Any) -> None:
        self.specs = specs

    def before_process(self, request: Any) -> Any:
        if (
            "image" in request.storage
            and isinstance(request.storage["image"], np.ndarray)
            and request.storage["image"].ndim in (2, 3)
        ):
            request.storage["tensor"] = self.to_bchw(request.storage["image"])

        if (
            "images" in request.storage
            and isinstance(request.storage["images"], list | tuple)
            and all(isinstance(x, np.ndarray) or x is None for x in request.storage["images"])
            and all(x.ndim in (2, 3) for x in request.storage["images"] if isinstance(x, np.ndarray))
        ):
            request.storage["tensors"] = [x if x is None else self.to_bchw(x) for x in request.storage["images"]]

        return request

    def to_bchw(self, image: np.ndarray) -> np.ndarray:
        image = np.float32(image)
        if hasattr(self.specs, "mean") and isinstance(self.specs.mean, np.ndarray):
            image = image - self.specs.mean
        if hasattr(self.specs, "stnd") and isinstance(self.specs.stnd, np.ndarray):
            image = image / self.specs.stnd

        # HWC to BCHW
        image = image[None, None] if image.ndim == 2 else np.transpose(image, (2, 0, 1))[None]
        return image
