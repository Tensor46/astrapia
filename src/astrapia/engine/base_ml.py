import logging
import pathlib
import platform
from abc import abstractmethod
from typing import Annotated, Any, Literal

import cpuinfo
import numpy as np
import onnxruntime as onnxrt
import pydantic

from astrapia.engine.base import Base


logger = logging.getLogger("BaseML")


def auto_onnxrt_providers(ignore_onnxrt_providers: list[str]) -> list[str]:
    def has_intel_on_linux() -> bool:
        info = cpuinfo.get_cpu_info()
        return platform.system() == "Linux" and "intel" in info.get("brand_raw", "").lower()

    # cpu is always default provider
    providers: list[str] = ["CPUExecutionProvider"]
    for provider in (
        "CoreMLExecutionProvider",  # when available, highest priority, usually on macOS
        "OpenVINOExecutionProvider",  # when available, higher priority than CPU -- only on linux
        "CUDAExecutionProvider",  # when available, higher priority than CPU and OpenVino
        "TensorrtExecutionProvider",  # when available, highest priority
    ):
        if provider not in onnxrt.get_available_providers():
            continue
        if provider == "OpenVINOExecutionProvider" and not has_intel_on_linux():
            continue
        if provider in ignore_onnxrt_providers:
            continue

        providers = [provider, *providers]
        if provider == "CoreMLExecutionProvider":  # Apple Silicon: rest can be ignored
            break
    return providers


class BaseML(Base):
    """Base class for ML inference processes supporting CoreML and ONNX Runtime.

    Handles model loading, input/output name extraction, and engine-specific calls.

    Specs:
        path_to_assets (Path): Base folder for model files.
        path_in_assets (str): Relative path/name of the model within assets.
        engine (Literal["coreml","onnxruntime"]): Backend to use.
        size (tuple[int,...]): Expected input dimensions.
        mean (np.ndarray|None): Optional per-channel mean for preprocessing.
        stnd (np.ndarray|None): Optional per-channel std for preprocessing.
        interpolation (Literal[1,2,3]): OpenCV resize flag.
        threshold (float): Decision threshold for outputs.
    """

    class Specs(Base.Specs, arbitrary_types_allowed=True, extra="ignore"):
        """Extended Specs for BaseML.

        Fields:
            path_to_assets (Path): Directory containing model files.
            path_in_assets (str): Model file name (without suffix).
            engine (str): "coreml" or "onnxruntime".
            size (tuple[int,...]): Input tensor shape.
            mean (np.ndarray|None): Mean normalization array.
            stnd (np.ndarray|None): Std normalization array.
            interpolation (int): Resize interpolation flag.
            threshold (float): Output threshold value.
            ignore_onnxrt_providers (list[str]): List of providers to ignore.
        """

        path_to_assets: pathlib.Path
        path_in_assets: str
        engine: Annotated[Literal["coreml", "onnxruntime"], pydantic.Field(frozen=True)]
        size: tuple[int, ...]
        mean: np.ndarray | None
        stnd: np.ndarray | None
        interpolation: Literal[1, 2, 3]
        threshold: float
        ignore_onnxrt_providers: Annotated[
            list[
                Literal[
                    "CoreMLExecutionProvider",
                    "CPUExecutionProvider",
                    "CUDAExecutionProvider",
                    "OpenVINOExecutionProvider",
                    "TensorrtExecutionProvider",
                ],
            ],
            pydantic.Field(default_factory=list),
        ]

        @pydantic.field_validator("mean", "stnd", mode="before")
        @classmethod
        def to_ndarray(cls, data: Any) -> np.ndarray | None:
            """Convert list/tuple to numpy array for mean/std fields."""
            if isinstance(data, list | tuple):
                data = np.squeeze(np.array(data, dtype=np.float32))
            return data

        @pydantic.field_serializer("mean", "stnd", when_used="json")
        def serialize_mean_stnd(self, data: np.ndarray) -> str:
            """Serialize mean/stnd numpy arrays to lists for JSON output."""
            return data.tolist() if isinstance(data, np.ndarray) else data

    __specs__ = Specs
    __inames__: tuple[str, ...] = ()
    __onames__: tuple[str, ...] = ()
    __onnxrt_providers__: Literal["AUTO", "CPU"] = "AUTO"

    def __init__(self, **kwargs) -> None:
        """Initialize specs and load the model."""
        super().__init__(**kwargs)
        self._load_model()

    @property
    def name(self) -> str:
        """Return the process name including engine info."""
        return f"{self.specs.name}: ({self.specs.version}, engine={self.specs.engine})"

    def _load_model(self) -> None:
        """Load the ML model from assets based on the specified engine.

        Raises:
            FileNotFoundError: If the model file or package is missing.
        """
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

            options = onnxrt.SessionOptions()
            options.enable_mem_pattern = True
            options.enable_mem_reuse = True
            options.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            if self.__onnxrt_providers__ == "AUTO":
                providers = auto_onnxrt_providers(self.specs.ignore_onnxrt_providers)
            else:
                providers = ["CPUExecutionProvider"]

            self.model = onnxrt.InferenceSession(path_to_model, sess_options=options, providers=providers)
            # get input and output names
            self.__inames__ = tuple(x.name for x in self.model.get_inputs())
            self.__onames__ = tuple(x.name for x in self.model.get_outputs())

    def _call_coreml(self, *args) -> dict[str, Any]:
        """Run inference on a CoreML model.

        Args:
            *args: Ordered inputs matching model input names.

        Returns:
            dict: Mapping of output names to prediction arrays.
        """
        output = self.model.predict(dict(zip(self.__inames__, args, strict=False)))
        return {name: output[name] for name in self.__onames__}

    def _call_onnxruntime(self, *args) -> dict[str, Any]:
        """Run inference on an ONNX Runtime session.

        Args:
            *args: Ordered inputs matching model input names.

        Returns:
            dict: Mapping of output names to prediction arrays.
        """
        output = self.model.run(list(self.__onames__), dict(zip(self.__inames__, args, strict=False)))
        return dict(zip(self.__onames__, output, strict=False))

    def process(self, *args) -> Any:
        """Dispatch inference call based on the selected engine.

        Args:
            *args: Model input tensors.

        Returns:
            Any: Prediction outputs.
        """
        if self.specs.engine == "coreml":
            output = self._call_coreml(*args)

        elif self.specs.engine == "onnxruntime":
            output = self._call_onnxruntime(*args)

        else:
            output = None

        return output

    @abstractmethod
    def default_response(self) -> Any:
        """Provide a default response structure for downstream handling."""
        ...
