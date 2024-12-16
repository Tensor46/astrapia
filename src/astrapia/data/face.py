from typing import Annotated, Literal

import cv2
import numpy as np
import pydantic

from astrapia.data.detection import BaseDetection
from astrapia.geometry.transform import similarity


class Face(BaseDetection):
    name: Annotated[Literal["FACE"], pydantic.Field(default="FACE")]
    points: np.ndarray

    __target_points__ = np.array([[0.39, 0.50], [0.61, 0.50], [0.50, 0.65], [0.50, 0.75]], dtype=np.float32)
    __target_corners__ = np.array([[0.25, 0.35], [0.75, 0.35], [0.75, 0.85], [0.25, 0.85]], dtype=np.float32)

    @pydantic.field_validator("points", mode="before")
    @classmethod
    def validate_points(cls, data: np.ndarray) -> np.ndarray:
        if isinstance(data, str):
            data = cls.decode(data)
        if isinstance(data, list | tuple):
            data = np.array(data, dtype=np.float32)
        if isinstance(data, np.ndarray):
            data = np.float32(data).reshape(-1, 2)

        if not (isinstance(data, np.ndarray) and data.shape in ((6, 2), (468, 2))):
            raise ValueError(f"{cls.__name__}: points ndarray of shape (6, 2) or (468, 2).")
        return data

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def iod(self) -> float:
        """Inter ocular distance."""
        return ((self.right_eye - self.left_eye) ** 2).sum().item() ** 0.5

    @property
    def eye_center(self) -> np.ndarray:
        """Eye center (x, y)."""
        return (self.right_eye + self.left_eye) / 2

    @property
    def right_eye(self) -> np.ndarray:
        """Right eye center (x, y)."""
        return self.index2point(self.right_eye_index)

    @property
    def right_eye_index(self) -> list[int]:
        """Right eye center indices."""
        return [0] if self.n_points == 6 else [243, 27, 130, 23]

    @property
    def left_eye(self) -> np.ndarray:
        """Left eye center (x, y)."""
        return self.index2point(self.left_eye_index)

    @property
    def left_eye_index(self) -> list[int]:
        """Left eye center indices."""
        return [1] if self.n_points == 6 else [463, 257, 359, 253]

    @property
    def nose_tip(self) -> np.ndarray:
        """Nose tip (x, y)."""
        return self.index2point(self.nose_tip_index)

    @property
    def nose_tip_index(self) -> list[int]:
        """Nose tip indices."""
        return [2] if self.n_points == 6 else [1, 4, 44, 274]

    @property
    def mouth(self) -> np.ndarray:
        """Mouth center (x, y)."""
        return self.index2point(self.mouth_index)

    @property
    def mouth_index(self) -> list[int]:
        """Mouth center indices."""
        return [3] if self.n_points == 6 else [0, 76, 17, 306]

    def index2point(self, index: tuple[int]):
        """Get points from indices."""
        return None if len(index) == 0 else self.points[index].mean(0)

    @property
    def source(self) -> np.ndarray:
        return np.stack((self.right_eye, self.left_eye, self.nose_tip, self.mouth), 0)

    def aligned(
        self,
        image: np.ndarray,
        max_side: int | None = None,
        force_max_side: bool = False,
        return_tm: bool = False,
    ) -> np.ndarray:
        """Align face with landmarks (eyes, nose-tip and mouth)."""
        source = np.stack((self.right_eye, self.left_eye, self.nose_tip, self.mouth), 0)
        reye, leye, *_ = self.__target_points__
        h = w = int(self.iod / (((reye - leye) ** 2).sum() ** 0.5))
        if max_side is not None and (max_side < h or force_max_side):
            h = w = int(max_side)

        target = self.__target_points__.copy()[: source.shape[0]]
        target[:, 0] *= w
        target[:, 1] *= h
        tm = similarity(source, target)

        aligend_image = cv2.warpAffine(image, tm[:2], (w, h))
        return (aligend_image, tm) if return_tm else aligend_image

    def crop_centered(self, image: np.ndarray, iod_multiplier: float = 4.0) -> np.ndarray:
        """Face crop with centered eyes."""
        x, y = self.eye_center.tolist()
        side = self.iod * iod_multiplier
        x1, y1, x2, y2 = list(map(int, (x - side / 2, y - side / 2, x + side / 2, y + side / 2)))
        return image[y1:y2, x1:x2]

    def aligned_corners(self) -> np.array:
        """Aligned corners."""
        # transformation matrix
        source = np.stack((self.right_eye, self.left_eye), 0)
        reye, leye, *_ = self.__target_points__
        scale = int(self.iod / (((reye - leye) ** 2).sum() ** 0.5))
        target = self.__target_points__.copy()[: source.shape[0]]
        target[:, 0] *= scale
        target[:, 1] *= scale
        tm = similarity(source, target)

        # convert target corners to image space
        target = self.__target_corners__.copy() * scale
        source = (np.linalg.inv(tm) @ np.concatenate((target, np.ones(target.shape[0])[:, None]), -1).T).T
        source = source[:, :2] / source[:, [2]]
        return source

    def annotate_aligned(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Annotate image."""
        image = image if inplace else image.copy()
        corners = np.round(self.aligned_corners()).astype(int)
        cv2.polylines(image, [corners.reshape(-1, 1, 2)], True, (236, 46, 36), 2)

        # add points
        radius = int(max(2, self.iod // 64))
        for x, y in (self.right_eye, self.left_eye, self.nose_tip):
            cv2.circle(image, (round(x), round(y)), radius, (16, 196, 146), -1)
        return image

    @pydantic.field_serializer("points", when_used="json")
    def serialize_points(self, data: np.ndarray) -> str:
        return self.encode(data)

    def __repr__(self) -> str:
        is_mesh = self.points.shape[0] == 468
        return f"Face({self.confidence:.4f}, box={self.box_cornerform}, iod={self.iod:.4f}, is_mesh={is_mesh})"
