import cv2
import numpy as np
import pydantic

from astrapia.data.base import BaseData


class Box(BaseData):
    """Bounding Box (center-x, center-y, width, height)."""

    x: float  # center-x
    y: float  # center-y
    w: float  # width
    h: float  # height

    @classmethod
    def from_cornerform(cls, x1, y1, x2, y2):
        return cls((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)

    @property
    def centerform(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    @property
    def numpy_centerform(self) -> np.ndarray:
        return np.float32(self.centerform)

    @property
    def cornerform(self) -> tuple[float, float, float, float]:
        return (self.x - 0.5 * self.w, self.y - 0.5 * self.h, self.x + 0.5 * self.w, self.y + 0.5 * self.h)

    @property
    def numpy_cornerform(self) -> np.ndarray:
        return np.float32(self.cornerform)


class Point(BaseData):
    """Point (x, y)."""

    x: float
    y: float

    def numpy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class BaseDetection(BaseData, arbitrary_types_allowed=True):
    label: str
    confidence: float
    box: np.ndarray

    @pydantic.field_validator("box", mode="before")
    @classmethod
    def validate_box(cls, data: np.ndarray) -> np.ndarray:
        if isinstance(data, str):
            data = cls.decode(data)
        if isinstance(data, list | tuple):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = np.float32(data).reshape(-1)

        if not (isinstance(data, np.ndarray) and data.size == 4):
            raise ValueError(f"{cls.__name__}: box ndarray of shape (4, ).")
        return data

    @property
    def box_cornerform(self) -> tuple[int, int, int, int]:
        return np.round(self.box.copy()).astype(int).tolist()

    def crop(self, image: np.ndarray, pad: float = 0.0) -> np.ndarray:
        """Detected box."""
        box = self.box.copy()
        if pad > 0:
            wp, hp = pad * (box[2] - box[0]), pad * (box[3] - box[1])
            box = (box[0] - wp / 2, box[1] - hp / 2, box[2] + wp / 2, box[3] + hp / 2)
        x1, y1, x2, y2 = map(int, box)
        return image[y1:y2, x1:x2]

    def annotate(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Annotate image."""
        image = image if inplace else image.copy()
        x1, y1, x2, y2 = self.box_cornerform
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(236, 46, 36), thickness=2)
        return image

    @pydantic.field_serializer("box", when_used="json")
    def serialize_box(self, data: np.ndarray) -> str:
        return self.encode(data)

    def __repr__(self) -> str:
        return f"Detection({self.confidence:.4f}, box={self.box_cornerform})"
