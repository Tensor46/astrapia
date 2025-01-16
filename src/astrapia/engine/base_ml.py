import logging
import pathlib
from abc import abstractmethod
from typing import Annotated, Any, Literal

import numpy as np
import pydantic

from astrapia.engine.base import Base


logger = logging.getLogger("BaseML")


class BaseML(Base):
    """Base ML process."""

    class Specs(Base.Specs, arbitrary_types_allowed=True, extra="ignore"):
        path_to_assets: pathlib.Path
        path_in_assets: str
        engine: Annotated[Literal["coreml", "onnxruntime"], pydantic.Field(frozen=True)]
        size: tuple[int, ...]
        mean: np.ndarray | None
        stnd: np.ndarray | None
        interpolation: Literal[1, 2, 3]
        threshold: float

        @pydantic.field_validator("mean", "stnd", mode="before")
        @classmethod
        def to_ndarray(cls, data: Any) -> np.ndarray | None:
            if isinstance(data, list | tuple):
                data = np.squeeze(np.array(data, dtype=np.float32))
            return data

        @pydantic.field_serializer("mean", "stnd", when_used="json")
        def serialize_mean_stnd(self, data: np.ndarray) -> str:
            return data.tolist() if isinstance(data, np.ndarray) else data

    __specs__ = Specs
    __inames__: tuple[str, ...] = ()
    __onames__: tuple[str, ...] = ()
    __onnx_providers__: Literal["AUTO", "CPU"] = "AUTO"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._load_model()

    @property
    def name(self) -> str:
        return f"{self.specs.name}: ({self.specs.version}, engine={self.specs.engine})"

    def _load_model(self) -> None:
        path_to_model = (self.specs.path_to_assets / self.specs.path_in_assets).with_suffix(".onnx")
        if self.specs.engine == "coreml":
            path_to_model = path_to_model.with_suffix(".mlmodel")

        if self.specs.engine == "coreml":  # load coreml model
            if not path_to_model.is_file():
                path_to_model = path_to_model.with_suffix(".mlpackage")
                if not path_to_model.is_dir():
                    logger.error(f"{self.__class__.__name__}: file | folder not found {path_to_model}")
                    raise FileNotFoundError(f"{self.__class__.__name__}: file | folder not found {path_to_model}")

            import coremltools as ct

            self.model = ct.models.MLModel(str(path_to_model))
            # get input and output names
            spec = self.model.get_spec()
            self.__inames__ = tuple(x.name for x in spec.description.input)
            self.__onames__ = tuple(x.name for x in spec.description.output)

        elif self.specs.engine == "onnxruntime":  # load onnxruntime model
            if not path_to_model.is_file():
                logger.error(f"{self.__class__.__name__}: file not found {path_to_model}")
                raise FileNotFoundError(f"{self.__class__.__name__}: file not found {path_to_model}")

            import onnxruntime as ort

            options = ort.SessionOptions()
            options.enable_mem_pattern = True
            options.enable_mem_reuse = True
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            providers = ["CPUExecutionProvider"]
            if self.__onnx_providers__ == "AUTO" and "TensorrtExecutionProvider" in ort.get_available_providers():
                providers = ["TensorrtExecutionProvider", *providers]
            if self.__onnx_providers__ == "AUTO" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", *providers]
            if self.__onnx_providers__ == "AUTO" and "CoreMLExecutionProvider" in ort.get_available_providers():
                providers = ["CoreMLExecutionProvider", *providers]
            self.model = ort.InferenceSession(path_to_model, sess_options=options, providers=providers)
            # get input and output names
            self.__inames__ = tuple(x.name for x in self.model.get_inputs())
            self.__onames__ = tuple(x.name for x in self.model.get_outputs())

    def _call_coreml(self, *args) -> dict[str, Any]:
        output = self.model.predict(dict(zip(self.__inames__, args, strict=False)))
        return {name: output[name] for name in self.__onames__}

    def _call_onnxruntime(self, *args) -> dict[str, Any]:
        output = self.model.run(list(self.__onames__), dict(zip(self.__inames__, args, strict=False)))
        return dict(zip(self.__onames__, output, strict=False))

    def process(self, *args) -> Any:
        if self.specs.engine == "coreml":
            output = self._call_coreml(*args)

        elif self.specs.engine == "onnxruntime":
            output = self._call_onnxruntime(*args)

        else:
            output = None

        return output

    @abstractmethod
    def default_response(self) -> Any: ...
