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
    """Tensor: Numpy ndarray with json serialization."""

    tensor: np.ndarray

    @pydantic.field_validator("tensor", mode="before")
    @classmethod
    def validate_tensor(cls, data: np.ndarray | str) -> np.ndarray:
        if isinstance(data, str):
            data = BaseData.decode(data)
        return data

    @pydantic.field_serializer("tensor", when_used="json")
    def serialize_tensor(self, data: np.ndarray) -> str:
        return BaseData.encode(data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.tensor.shape


class ImageTensor(BaseTensor):
    detections: Annotated[list[BaseDetection | Face], pydantic.Field(default=[])]

    @pydantic.field_validator("tensor", mode="before")
    @classmethod
    def validate_tensor(cls, data: np.ndarray | pathlib.Path | str) -> np.ndarray:
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
        if isinstance(data, list | tuple):
            data = [Face(**x) if isinstance(x, dict) and x["label"] == "FACE" else x for x in data]
            data = [BaseDetection(**x) if isinstance(x, dict) and x["label"] != "FACE" else x for x in data]
        return data

    @property
    def pil_image(self) -> ImPIL.Image:
        return ImPIL.fromarray(self.tensor)

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    def to_gray(self, inplace: bool = False) -> np.ndarray:
        gray = cv2.cvtColor(self.tensor, cv2.COLOR_RGB2GRAY)
        if inplace:
            self.tensor = gray
        return gray

    def to_bchw(self, gray: bool = False) -> np.ndarray:
        return (self.to_gray()[None, None] if gray else np.transpose(self.tensor.copy(), (2, 0, 1)))[None]

    def annotate(self, **kwargs) -> np.ndarray:
        """Annotate image."""
        image = self.tensor.copy()
        for detection in self.detections:
            detection.annotate(image, inplace=True, **kwargs)
        return image

    def resize(self, size: tuple[int, int], interpolation: int = 3) -> np.ndarray:
        # resize
        #   size: (height x width)
        #   interpolation: 1 is INTER_LINEAR, 2 is INTER_CUBIC, 3 is INTER_AREA
        return cv2.resize(self.tensor, size[::-1], interpolation=interpolation)

    def resize_with_pad(self, size: tuple[int, int], interpolation: int = 3) -> tuple[np.ndarray, float]:
        # resize and keep aspect ratio
        #   size: (height x width)
        #   interpolation: 1 is INTER_LINEAR, 2 is INTER_CUBIC, 3 is INTER_AREA
        scale = size[0] / self.shape[0]
        if any(int(tsz * scale) > sz for tsz, sz in zip(self.shape, size, strict=False)):
            scale = size[1] / self.shape[1]
        size_new = tuple(map(int, map(round, (self.shape[0] * scale, self.shape[1] * scale))))

        tensor = cv2.resize(self.tensor, size_new[::-1], interpolation=interpolation)
        canvas = np.zeros(list(size) + ([self.shape[-1]] if len(self.shape) == 3 else []), tensor.dtype)
        canvas[: size_new[0], : size_new[1]] = tensor
        return canvas, scale

    def load_detections(self, path: pathlib.Path) -> None:
        with open(path) as txt:
            detections = yaml.safe_load(txt.read())["detections"]

        self.detections += ImageTensor.validate_detections(detections)

    def save_detections(self, path: pathlib.Path) -> None:
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"{self.__class__.__name__}.save_detections: path must be pathlib.Path object.")
        if path.suffix != ".yaml":
            raise TypeError(f"{self.__class__.__name__}.save_detections: path must have '.yaml' as suffix.")

        for detection in self.detections:
            detection.clear_storage()
        with open(path, "w") as txt:
            txt.write(self.model_dump_json(exclude={"storage", "tensor"}, indent=2))
