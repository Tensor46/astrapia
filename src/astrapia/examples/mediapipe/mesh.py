from typing import Any, Literal

import numpy as np
import pydantic

from astrapia import assets
from astrapia.callbacks.base import BaseCallback
from astrapia.callbacks.to_bchw import ToBCHW
from astrapia.data.face import Face
from astrapia.data.tensor import ImageTensor
from astrapia.engine.base_ml import BaseMLProcess


class PrepareImages(BaseCallback):
    def __init__(self, specs: pydantic.BaseModel) -> None:
        self.specs = specs

    def before_process(self, request: ImageTensor) -> Any:
        request.storage["images"] = []
        request.storage["tms"] = []
        for detection in request.detections:
            if isinstance(detection, Face):
                image, tm = detection.aligned(
                    request.tensor,
                    max_side=max(self.specs.size),
                    force_max_side=True,
                    return_tm=True,
                )
                request.storage["images"].append(image)
                request.storage["tms"].append(tm)
            else:
                request.storage["images"].append(None)
                request.storage["tms"].append(None)
        return request


class Process(BaseMLProcess):
    class Specs(BaseMLProcess.Specs):
        name: Literal["MediaPipe-Mesh"]
        version: Literal["mesh"]

    __specs__ = Specs
    __requests__ = (ImageTensor,)
    __response__ = ImageTensor

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__callbacks__ += (PrepareImages(self.specs), ToBCHW(self.specs))

    @classmethod
    def load_mesh(cls, engine: Literal["coreml", "onnxruntime"]):
        specs = assets.load_yaml(assets.PATH_TO_ASSETS / "configs/mediapipe_mesh.yaml")
        specs["path_to_assets"] = assets.PATH_TO_ASSETS
        specs["engine"] = engine
        return cls(**specs)

    def process(self, request: Any) -> Any:
        for tensor, tm_image2crop, face in zip(
            request.storage["tensors"],
            request.storage["tms"],
            request.detections,
            strict=False,
        ):
            if tensor is None or tm_image2crop is None:
                continue

            # infer model
            output = super().process(tensor)
            output = {key: np.squeeze(value) for key, value in output.items()}
            score, points_on_crop = output["scores"], output["points"][..., :2]
            points_on_crop *= max(tensor.shape[2:])
            # not reliable
            if score < self.specs.threshold:
                continue
            # project points on crop to image
            points_on_crop = np.concatenate((points_on_crop, np.ones(points_on_crop.shape[0])[:, None]), -1)
            points_on_image = (np.linalg.inv(tm_image2crop) @ points_on_crop.T).T
            points_on_image = points_on_image[:, :2] / points_on_image[:, [2]]
            face.points = points_on_image

        request.clear()
        return request

    def default_response(self, **kwargs) -> Any:
        raise NotImplementedError()
