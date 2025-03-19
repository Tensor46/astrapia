from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pydantic

from astrapia import assets
from astrapia.callbacks.base import BaseCallback
from astrapia.callbacks.to_bchw import ToBCHW
from astrapia.data.face import Face
from astrapia.data.tensor import ImageTensor
from astrapia.engine.base_ml import BaseML
from astrapia.examples.mediapipe.mesh import Algorithm as AlgoMesh
from astrapia.utils.nms_fns import nms_numba


class PrepareImages(BaseCallback):
    def __init__(self, specs: pydantic.BaseModel) -> None:
        self.specs = specs

    def before_process(self, request: ImageTensor) -> Any:
        request.storage["images"] = []
        request.storage["scales"] = []
        request.storage["sizes"] = []
        for size in [self.specs.size, *self.specs.extra.sizes]:
            image, scale = request.resize_with_pad(size)
            request.storage["images"].append(image)
            request.storage["scales"].append(scale)
            request.storage["sizes"].append(size)
        return request


class AddMesh(BaseCallback):
    def __init__(self, specs: pydantic.BaseModel) -> None:
        self.specs = specs
        self.mesh_model = AlgoMesh.load_mesh(engine=self.specs.engine)

    def after_process(self, request: ImageTensor) -> Any:
        self.mesh_model(request)
        return request


class Algorithm(BaseML):
    class Specs(BaseML.Specs):
        name: Literal["MediaPipe-Detector"]
        version: Literal["short", "long"]

    __specs__ = Specs
    __requests__ = (ImageTensor,)
    __response__ = ImageTensor

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__callbacks__ += (PrepareImages(self.specs), ToBCHW(self.specs))
        if self.specs.extra.add_mesh:
            self.__callbacks__ += (AddMesh(self.specs),)

    def add_extra_size(self, size: tuple[int, int]) -> None:
        self.specs.extra.sizes.add(size)

    def _compute_anchors(self, size: tuple[int, int]) -> np.ndarray:
        @lru_cache(maxsize=4)
        def _compute_anchors(size: tuple[int, int], strides: tuple[int, ...]):
            h, w = size
            anchors: list[np.ndarray] = []
            for st in np.sort(np.unique(strides)).tolist():
                repeats = (np.array(strides) == st).sum() * 2
                nh = h // st
                nw = w // st
                ys = np.linspace(0.5, nh - 0.5, nh) / nh
                xs = np.linspace(0.5, nw - 0.5, nw) / nw
                yxs = np.stack(np.meshgrid(ys, xs), axis=-1).reshape(-1, 2)
                anchors.append(yxs.repeat(repeats, axis=0))
            return np.concatenate(anchors).astype(np.float32)

        return _compute_anchors(size, self.specs.extra.strides)

    @classmethod
    def load_short(cls, engine: Literal["coreml", "onnxruntime"], **kwargs):
        specs = assets.load_yaml(assets.PATH_TO_ASSETS / "configs/mediapipe_short.yaml")
        specs["path_to_assets"] = assets.PATH_TO_ASSETS
        specs["engine"] = engine
        return cls(**specs, **kwargs)

    @classmethod
    def load_long(cls, engine: Literal["coreml", "onnxruntime"], **kwargs):
        specs = assets.load_yaml(assets.PATH_TO_ASSETS / "configs/mediapipe_long.yaml")
        specs["path_to_assets"] = assets.PATH_TO_ASSETS
        specs["engine"] = engine
        return cls(**specs, **kwargs)

    def process(self, request: Any) -> Any:
        all_scores, all_boxes, all_points = None, None, None
        for tensor, scale, size in zip(
            request.storage["tensors"],
            request.storage["scales"],
            request.storage["sizes"],
            strict=False,
        ):
            # infer model
            output = super().process(tensor)
            output = {key: np.squeeze(value) for key, value in output.items()}
            anchors = self._compute_anchors(size)

            # decode points
            scores, regression = output["scores"], output["regression"]
            num_points = regression.shape[-1] // 2
            regression = regression.reshape(-1, num_points, 2) / max(size)
            regression += anchors[:, None, :]
            regression[:, 1] -= anchors

            # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
            # split into boxes and points
            boxes = np.zeros_like(regression[:, :2])
            boxes[:, 0] = regression[:, 0] - regression[:, 1] / 2
            boxes[:, 1] = regression[:, 0] + regression[:, 1] / 2
            points = regression[:, 2:]
            boxes = np.clip(boxes * max(size), 0, None)
            boxes = boxes.reshape(-1, 4)
            points = np.clip(points * max(size), 0, None)
            # thresholding
            threshold = self.specs.threshold if size == self.specs.size else self.specs.extra.threshold_extra_sizes
            valid = scores >= threshold
            scores, boxes, points = (x[valid] for x in (scores, boxes, points))

            # scale to image size
            boxes /= scale
            points /= scale

            # aggregate all
            all_scores = scores.copy() if all_scores is None else np.concatenate((all_scores, scores))
            all_boxes = boxes.copy() if all_boxes is None else np.concatenate((all_boxes, boxes))
            all_points = points.copy() if all_points is None else np.concatenate((all_points, points))

        # highest score first
        sort = np.flip(np.argsort(all_scores))
        all_scores, all_boxes, all_points = (x[sort] for x in (all_scores, all_boxes, all_points))
        # remove invalid area
        if self.specs.extra.min_area is not None:
            valid = (((all_boxes[..., 2:] - all_boxes[..., :2]) ** 2).sum(-1) ** 0.5) >= self.specs.extra.min_area
            all_scores, all_boxes, all_points = (x[valid] for x in (all_scores, all_boxes, all_points))
        # nms
        valid = nms_numba(all_boxes, all_scores, self.specs.extra.threshold_iou) == 1
        all_scores, all_boxes, all_points = (x[valid] for x in (all_scores, all_boxes, all_points))

        # add all detections
        for confidence, box, points in zip(all_scores.tolist(), all_boxes, all_points, strict=False):
            request.detections.append(Face(label="FACE", confidence=confidence, box=box, points=points))
        return request

    def default_response(self, **kwargs) -> Any:
        # Since request is the response!
        raise NotImplementedError()

    def default_specs(self, **kwargs) -> dict[str, Any]:
        kwargs.setdefault("version", "short")
        kwargs.setdefault("sizes", set())
        kwargs.setdefault("add_mesh", False)
        kwargs["strides"] = (8, 16, 16, 16) if kwargs["version"] == "short" else (16, 32, 32, 32)
        kwargs.setdefault("threshold_iou", 0.3)
        kwargs.setdefault("threshold_extra_sizes", 0.8)
        kwargs.setdefault("min_area", 12)
        return kwargs
