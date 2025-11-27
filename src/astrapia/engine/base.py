import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Annotated, Any

import numpy as np
import pydantic

from astrapia import assets
from astrapia.callbacks.base import BaseCallback
from astrapia.data.base import BaseDataWithExtra
from astrapia.utils import timer


logger = logging.getLogger("Base")


class Base(ABC):
    """Abstract base for request-response processing pipelines.

    Defines a `Specs` inner model for configuration, callback hooks,
    and common utilities (sigmoid, softmax, YAML loading).
    """

    class Specs(BaseDataWithExtra):
        """Specification model for a processing instance.

        Fields:
            name (str): Identifier for this process.
            version (str): Lowercased version string.
            clear_storage_on_exit (bool): If True, clear request/response storage after processing.
            extra (Extra): Any additional keyword fields.
        """

        name: str
        version: Annotated[str, pydantic.StringConstraints(strip_whitespace=True, to_lower=True)]
        clear_storage_on_exit: Annotated[bool, pydantic.Field(default=True, frozen=True)]

    __callbacks__: tuple[BaseCallback, ...] = ()
    __extra__ = None
    __requests__: tuple[Any, ...] = ()
    __response__: Any = None
    __specs__: pydantic.BaseModel = Specs

    def __init__(self, **kwargs) -> None:
        """Initialize with `Specs`, applying any default_specs adjustments."""
        if not issubclass(self.__specs__, BaseDataWithExtra):
            raise TypeError(f"{self.__class__.__name__}: __specs__ must be inherited from BaseSpecs.")
        self.specs = self.__specs__(**self.default_specs(**kwargs))

    @property
    def name(self) -> str:
        """Return formatted name and version."""
        return f"{self.specs.name}: ({self.specs.version})"

    @classmethod
    def from_yaml(cls, path_to_assets: pathlib.Path, path_to_yaml: pathlib.Path):
        """Create an instance from YAML specs.

        Args:
            path_to_assets (Path | None): Base directory for asset files.
            path_to_yaml (Path): Path to the YAML spec file.

        Returns:
            Base: New instance with loaded specs.
        """
        if not (isinstance(path_to_assets, pathlib.Path) or path_to_assets is None):
            logger.error(f"{cls.__name__}: path_to_assets must be None | pathlib.Path - {path_to_assets}")
            raise TypeError(f"{cls.__name__}: path_to_assets must be None | pathlib.Path - {path_to_assets}")

        if not isinstance(path_to_yaml, pathlib.Path):
            logger.error(f"{cls.__name__}: path_to_yaml must be pathlib.Path - {path_to_yaml}")
            raise TypeError(f"{cls.__name__}: path_to_yaml must be pathlib.Path - {path_to_yaml}")

        specs = assets.load_yaml(path_to_yaml)
        if path_to_assets is not None:
            specs["path_to_assets"] = path_to_assets

        return cls(**specs)

    def validate_request(self, request: Any) -> None:
        """Ensure the request is of an acceptable type.

        Raises:
            TypeError: If `request` is not an instance of allowed types.
        """
        if not isinstance(request, self.__requests__):
            logger.error(f"{self.specs.name}: invalid type for request.")
            raise TypeError(f"{self.specs.name}: invalid type for request.")

    def __call__(self, *args) -> Any:
        """Invoke processing on one or more requests.

        - Single arg: returns a response.
        - Multiple args: returns a list of responses.
        """
        if len(args) == 0:
            logger.error(f"{self.specs.name}: missing request.")
            raise TypeError(f"{self.specs.name}: missing request.")
        if len(args) == 1:
            with timer.Timer(name=self.name):
                return self.__process__(*args)
        return [self.__process__(request=arg) for arg in args]

    def __process__(self, request: Any) -> Any:
        """Validate, invoke callbacks, process, and clear storage as configured.

        Args:
            request (Any): Input object for processing.

        Returns:
            Any: The processed response.
        """
        self.validate_request(request)
        # callbacks - before
        for callback in self.__callbacks__:
            callback.before_process(request)
        # process
        response = self.process(request)
        # callbacks - after
        for callback in self.__callbacks__:
            callback.after_process(response)
        # clear storage
        if self.specs.clear_storage_on_exit and hasattr(request, "clear_storage"):
            request.clear_storage()
        if self.specs.clear_storage_on_exit and hasattr(response, "clear_storage"):
            response.clear_storage()
        return response

    @abstractmethod
    def process(self, request: Any) -> Any:
        """Perform the core processing logic. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def default_response(self) -> Any:
        """Return a default response structure. Must be implemented by subclasses."""
        ...

    def default_specs(self, **kwargs) -> dict[str, Any]:
        """Hook to adjust raw kwargs before spec validation.

        Returns:
            dict: Possibly modified kwargs for Specs.
        """
        return kwargs

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid activation on `x`.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid of input.
        """
        pos = x > 0
        neg = ~pos
        x[pos] = 1 / (1 + np.exp(-x[pos]))
        x[neg] = np.exp(x[neg]) / (1 + np.exp(x[neg]))
        return x

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute the softmax of an array along a specified axis in a numerically stable way.

        Args:
            x (np.ndarray): Input array.
            axis (int, optional): Axis along which to apply softmax. Defaults to -1.

        Returns:
            np.ndarray: An array of the same shape as `x`, where the values along the
                        specified axis sum to 1.
        """
        xexp = np.exp(x - x.max(axis, keepdims=True))
        return xexp / xexp.sum(axis, keepdims=True)

    def __repr__(self) -> str:
        return f"{self.specs.name}{self.specs.model_dump_json(exclude={'clear_storage_on_exit'}, indent=2)}"
