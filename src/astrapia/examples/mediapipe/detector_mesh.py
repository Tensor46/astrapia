from typing import Annotated, Any, Literal

import pydantic

from astrapia.data.tensor import ImageTensor
from astrapia.engine.base_ml import BaseProcess
from astrapia.examples.mediapipe import detector, mesh


class Process(BaseProcess):
    class Specs(BaseProcess.Specs):
        name: Annotated[Literal["MediaPipe"], pydantic.Field(default="MediaPipe", frozen=True)]
        version: Annotated[Literal["short", "long"], pydantic.Field(default="short", frozen=True)]
        do_mesh: bool = False
        engine: Annotated[Literal["coreml", "onnxruntime"], pydantic.Field(default="onnxruntime", frozen=True)]

    __specs__ = Specs
    __requests__ = (ImageTensor,)
    __response__ = ImageTensor

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.specs.version == "short":
            self.detector = detector.Process.load_short(self.specs.engine)
        else:
            self.detector = detector.Process.load_long(self.specs.engine)
        self.mesh = mesh.Process.load_mesh(self.specs.engine) if self.specs.do_mesh else None

    def add_extra_size(self, size: tuple[int, int]) -> None:
        self.detector.add_extra_size(size)

    def process(self, request: Any) -> Any:
        self.detector(request)
        if self.mesh is not None:
            self.mesh(request)

        return request

    def default_response(self, **kwargs) -> Any:
        # Since request is the response!
        raise NotImplementedError()
