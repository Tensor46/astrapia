from typing import Annotated

import cv2
import numpy as np
import pydantic

from astrapia.data.base import BaseData


class BaseDetection(BaseData, arbitrary_types_allowed=True):
    """Represents a single object detection result.

    Attributes:
        label (str): Class label for the detection.
        confidence (float): Confidence score [0.0-1.0].
        box (np.ndarray): Bounding box in corner form [x1, y1, x2, y2].
        embedding (np.ndarray | None): Optional feature embedding (1D array).
        points (np.ndarray | None): Optional keypoints (Nx2 array).
    """

    label: str
    confidence: float
    box: np.ndarray
    embedding: Annotated[np.ndarray | None, pydantic.Field(default=None)]
    points: Annotated[np.ndarray | None, pydantic.Field(default=None)]

    @pydantic.field_validator("box", "embedding", mode="before")
    @classmethod
    def validate_1d_ndarray(cls, data: np.ndarray) -> np.ndarray:
        """Ensure box and embedding are 1D float32 arrays. Accepts base64 strings, lists/tuples, or ndarrays."""
        if isinstance(data, str):
            data = cls.decode(data)
        if isinstance(data, list | tuple):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = np.float32(data).reshape(-1)
        return data

    @pydantic.field_validator("points", mode="before")
    @classmethod
    def validate_points(cls, data: np.ndarray) -> np.ndarray:
        """Ensure points are Nx2 float32 arrays. Accepts base64 strings, lists/tuples, or ndarrays."""
        if isinstance(data, str):
            data = cls.decode(data)
        if isinstance(data, list | tuple):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = np.float32(data).reshape(-1, 2)
        return data

    @property
    def centerform(self) -> np.ndarray:
        """Return the bounding box in [cx, cy, w, h] format."""
        return self.cornerform_to_centerform(self.cornerform)

    @property
    def cornerform(self) -> np.ndarray:
        """Return the bounding box in [x1, y1, x2, y2] format."""
        return self.box.copy()

    @property
    def corners(self) -> tuple[tuple[float, float], ...]:
        """Return the four corner coordinates as: ((x1, y1), (x2, y1), (x2, y2), (x1, y2))."""
        x1, y1, x2, y2 = self.cornerform.tolist()
        return (
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        )

    def crop(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Crop the image to the detection box (optionally scaled).

        Args:
            image (np.ndarray): Input image array (HxWxC).
            scale (float): Scale factor for the box size.
        """
        x1, y1, x2, y2 = np.round(self.crop_xyxy(scale=scale)).astype(int).tolist()
        return image[y1:y2, x1:x2]

    def crop_xyxy(self, scale: float = 1.0) -> np.ndarray:
        """Compute a scaled cornerform box from the centerform.

        Args:
            scale (float): Scale factor for width and height.
        """
        x, y, w, h = self.centerform.tolist()
        w *= scale
        h *= scale
        return self.centerform_to_cornerform((x, y, w, h))

    def annotate(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Draw the bounding box on the image.

        Args:
            image (np.ndarray): Input image array (HxWxC).
            inplace (bool): If False, operates on a copy.
        """
        image = image if inplace else image.copy()
        x1, y1, x2, y2 = np.round(self.cornerform).astype(int).tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(236, 46, 36), thickness=2)
        return image

    @staticmethod
    def cornerform_to_centerform(box: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h].

        Args:
            box (np.ndarray): Cornerform array of length 4.
        """
        x1, y1, x2, y2 = np.float32(box)
        return np.float32([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

    @staticmethod
    def centerform_to_cornerform(box: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2].

        Args:
            box (np.ndarray): Centerform array of length 4.
        """
        x, y, w, h = np.float32(box)
        return np.float32([x - w / 2, y - h / 2, x + w / 2, y + h / 2])

    @pydantic.field_serializer("box", "embedding", "points", when_used="json")
    def serialize_ndarray(self, data: np.ndarray | None) -> str | None:
        """Serialize arrays to base64-encoded strings for JSON output.

        Returns None if data is None.
        """
        return data if data is None else self.encode(data)

    def __repr__(self) -> str:
        box = np.round(self.cornerform).astype(int).tolist()
        return f"Detection({self.confidence:.4f}, box={box})"
