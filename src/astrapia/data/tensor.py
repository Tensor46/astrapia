import os
import pathlib
from typing import Annotated

import cv2
import numpy as np
import pydantic
import yaml
from PIL import Image as ImPIL

from astrapia.data.base import BaseData
from astrapia.data.detection import BaseDetection
from astrapia.data.face import Face


class BaseTensor(BaseData, arbitrary_types_allowed=True):
    """Wrapper for a numpy array with JSON-serializable support.

    Attributes:
        tensor (np.ndarray): The underlying array data.
    """

    tensor: np.ndarray

    @pydantic.field_validator("tensor", mode="before")
    @classmethod
    def validate_tensor(cls, data: np.ndarray | str) -> np.ndarray:
        """Ensure `tensor` is loaded as an ndarray.

        Args:
            data (np.ndarray | str): Raw array or base64 string.

        Returns:
            np.ndarray: Decoded or original array.

        Raises:
            ValueError: If the decoded data is invalid.
        """
        if isinstance(data, str):
            data = BaseData.decode(data)
        return data

    @pydantic.field_serializer("tensor", when_used="json")
    def serialize_tensor(self, data: np.ndarray) -> str:
        """Convert `tensor` to a base64 string for JSON output.

        Returns:
            str: Encoded data.
        """
        return BaseData.encode(data)

    @property
    def size(self) -> tuple[int, ...]:
        """Return the size of the tensor."""
        return self.tensor.shape


class ImageTensor(BaseTensor):
    """Image container with optional detections.

    Attributes:
        tensor (np.ndarray): HxWxC image array (RGB or grayscale).
        detections (list[BaseDetection | Face]): Detected objects on the image.
    """

    detections: Annotated[list[BaseDetection | Face], pydantic.Field(default=[])]

    @pydantic.field_validator("tensor", mode="before")
    @classmethod
    def validate_tensor(cls, data: np.ndarray | pathlib.Path | str) -> np.ndarray:
        """Load or decode an image into an ndarray.

        Args:
            data (np.ndarray | Path | str): Array, filepath, or base64.

        Returns:
            np.ndarray: RGB or grayscale array.

        Raises:
            ValueError: If the result is not a valid 2D/3D image array.
        """
        if isinstance(data, pathlib.Path):
            data = cv2.cvtColor(cv2.imread(str(data), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if isinstance(data, str):
            data = cv2.cvtColor(cv2.imread(data, 1), 4) if os.path.isfile(data) else BaseTensor.decode(data)

        if not (isinstance(data, np.ndarray) and (data.ndim == 2 or (data.ndim == 3 and data.shape[-1] in (1, 3)))):
            raise ValueError(f"{cls.__name__}: tensor must be 2/3 dimensional ndarray.")
        return data

    @pydantic.field_validator("detections", mode="before")
    @classmethod
    def validate_detections(cls, data: list[BaseDetection | Face]) -> np.ndarray:
        """Ensure each detection is an instance of BaseDetection or Face.

        Args:
            data (list): Raw detection dicts or objects.

        Returns:
            list: Validated detection objects.
        """
        if isinstance(data, list | tuple):
            data = [Face(**x) if isinstance(x, dict) and x["label"] == "FACE" else x for x in data]
            data = [BaseDetection(**x) if isinstance(x, dict) and x["label"] != "FACE" else x for x in data]
        return data

    @property
    def pil_image(self) -> ImPIL.Image:
        """Convert `tensor` to a PIL Image."""
        return ImPIL.fromarray(self.tensor)

    @property
    def height(self) -> int:
        return self.size[0]

    @property
    def width(self) -> int:
        return self.size[1]

    def to_gray(self, inplace: bool = False) -> np.ndarray:
        """Convert image to grayscale.

        Args:
            inplace (bool): If True, update `tensor` in place.

        Returns:
            np.ndarray: Grayscale image.
        """
        gray = cv2.cvtColor(self.tensor, cv2.COLOR_RGB2GRAY)
        if inplace:
            self.tensor = gray
        return gray

    def to_bchw(self, gray: bool = False) -> np.ndarray:
        """Convert to a BxCxHxW tensor batch.

        Args:
            gray (bool): If True, use grayscale.

        Returns:
            np.ndarray: 4D tensor with batch dimension.
        """
        return (self.to_gray()[None, None] if gray else np.transpose(self.tensor.copy(), (2, 0, 1)))[None]

    def annotate(self, **kwargs) -> np.ndarray:
        """Draw all detections on a copy of the image.

        Returns:
            np.ndarray: Annotated image.
        """
        image = self.tensor.copy()
        for detection in self.detections:
            detection.annotate(image, inplace=True, **kwargs)
        return image

    def resize(self, size: tuple[int, int], interpolation: int = 3) -> np.ndarray:
        """
        Resize while ignoring aspect ratio.

        Args:
            size (h, w): New size.
            interpolation (int): OpenCV interpolation flag.
                1 is INTER_LINEAR, 2 is INTER_CUBIC, 3 is INTER_AREA

        Returns:
            np.ndarray: Resized image.
        """
        return cv2.resize(self.tensor, size[::-1], interpolation=interpolation)

    def resize_with_pad(self, size: tuple[int, int], interpolation: int = 3) -> tuple[np.ndarray, float]:
        """Resize with aspect-ratio padding to fit `size`.

        Args:
            size (h, w): Target canvas size.
            interpolation (int): OpenCV interpolation flag.
                1 is INTER_LINEAR, 2 is INTER_CUBIC, 3 is INTER_AREA

        Returns:
            (padded_image, scale): The new image and scale factor.
        """
        scale = size[0] / self.size[0]
        if any(int(tsz * scale) > sz for tsz, sz in zip(self.size, size, strict=False)):
            scale = size[1] / self.size[1]
        size_new = tuple(round(s * scale) for s in self.size[:2])

        tensor = cv2.resize(self.tensor, size_new[::-1], interpolation=interpolation)
        canvas = np.zeros(list(size) + ([self.size[-1]] if len(self.size) == 3 else []), tensor.dtype)
        canvas[: size_new[0], : size_new[1]] = tensor
        return canvas, scale

    def load_detections(self, path: pathlib.Path) -> None:
        """Load detections from a YAML file and append to `detections`.

        Args:
            path (Path): YAML file containing {"detections": [...]}.
        """
        with open(path) as txt:
            detections = yaml.safe_load(txt.read())["detections"]

        self.detections += ImageTensor.validate_detections(detections)

    def save_detections(self, path: pathlib.Path) -> None:
        """Save current detections to a YAML file.

        Args:
            path (Path): Must have a .yaml suffix.

        Raises:
            TypeError: If path is invalid or not .yaml.
        """
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"{self.__class__.__name__}.save_detections: path must be pathlib.Path object.")
        if path.suffix != ".yaml":
            raise TypeError(f"{self.__class__.__name__}.save_detections: path must have '.yaml' as suffix.")

        for detection in self.detections:
            detection.clear_storage()
        with open(path, "w") as txt:
            txt.write(self.model_dump_json(exclude={"storage", "tensor"}, indent=2))
