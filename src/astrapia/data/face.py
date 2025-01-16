from typing import Annotated, Literal

import cv2
import numpy as np
import pydantic

from astrapia.data.detection import BaseDetection
from astrapia.geometry import transform


class Face(BaseDetection):
    name: Annotated[Literal["FACE"], pydantic.Field(default="FACE")]

    __target_corners__ = np.array([[0.25, 0.35], [0.75, 0.35], [0.75, 0.85], [0.25, 0.85]], dtype=np.float32)
    __target_points__ = np.array([[0.39, 0.50], [0.61, 0.50], [0.50, 0.65], [0.50, 0.75]], dtype=np.float32)

    @pydantic.field_validator("box", "embedding", mode="before")
    @classmethod
    def validate_1d_ndarray(cls, data: np.ndarray) -> np.ndarray:
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
    def iod_approximate(self) -> float:
        """Approximate IOD better for off angle faces."""
        points = self.source[:3]
        return (((points[:, None] - points[None]) ** 2).sum(-1) ** 0.5).max()

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

    def aligned_face(
        self,
        image: np.ndarray,
        side: int | None = None,
        pad: float = 0.0,
        allow_smaller_side: bool = True,
    ) -> np.ndarray:
        """Align face with landmarks (eyes, nose-tip and mouth)."""
        source = np.stack((self.right_eye, self.left_eye, self.nose_tip, self.mouth), 0)
        reye, leye, *_ = self.__target_points__
        h = w = int(self.iod_approximate / (((reye - leye) ** 2).sum() ** 0.5))
        if side is not None:
            h = w = h if allow_smaller_side and side > h else side

        target = self.__target_points__.copy()[: source.shape[0]]
        if pad != 0.0:
            target += pad / 2
            target /= 1 + pad
        target[:, 0] *= w
        target[:, 1] *= h
        tm = transform.similarity(source, target)
        aligend_image, _ = transform.source2target_converter(image, None, size_hxw=(h, w), tmat=tm)
        self.storage["tmat"] = tm
        return aligend_image

    def face_crop(self, image: np.ndarray, iod_multiplier: float = 4.0) -> np.ndarray:
        """Face crop with eyes at the center."""
        x, y = self.eye_center.tolist()
        side = self.iod_approximate * iod_multiplier
        x1, y1, x2, y2 = (int(xy) for xy in (x - side / 2, y - side / 2, x + side / 2, y + side / 2))
        return image[y1:y2, x1:x2]

    def aligned_corners(self) -> np.array:
        """Aligned corners."""
        # transformation matrix
        source = np.stack((self.right_eye, self.left_eye), 0)
        reye, leye, *_ = self.__target_points__
        scale = int(self.iod / (((reye - leye) ** 2).sum() ** 0.5))
        target = self.__target_points__.copy()[: source.shape[0]]
        tm = transform.similarity(source, target * scale)

        # convert target corners to image space
        target = self.__target_corners__.copy() * scale
        _, source = transform.source2target_converter(None, points=target, tmat=tm, invert=True)
        return source

    def annotate(self, image: np.ndarray, inplace: bool = False, all_points: bool = False) -> np.ndarray:
        """Annotate image."""
        image = super().annotate(image, inplace=inplace)
        # add points
        radius = int(max(2, self.iod_approximate // 64))
        for x, y in self.points if all_points else (self.right_eye, self.left_eye, self.nose_tip):
            cv2.circle(image, (round(x), round(y)), radius, (16, 196, 146), -1)
        return image

    def __repr__(self) -> str:
        is_mesh = self.points.shape[0] == 468
        box = np.round(self.cornerform).astype(int).tolist()
        return f"Face({self.confidence:.4f}, box={box}, iod={self.iod:.4f}, is_mesh={is_mesh})"
