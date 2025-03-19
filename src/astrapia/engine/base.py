import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Annotated, Any

import numpy as np
import pydantic

from astrapia import assets
from astrapia.callbacks.base import BaseCallback
from astrapia.utils import timer


logger = logging.getLogger("Base")


class Extra(pydantic.BaseModel, extra="allow"): ...


class Base(ABC):
    """Base process."""

    class Specs(pydantic.BaseModel, extra="ignore"):
        name: str
        version: Annotated[str, pydantic.StringConstraints(strip_whitespace=True, to_lower=True)]
        clear_storage: Annotated[bool, pydantic.Field(default=True, frozen=True)]
        extra: Annotated[Extra, pydantic.Field(default=Extra())]

        @pydantic.model_validator(mode="before")
        @classmethod
        def extras(cls, data: dict[str, Any]) -> dict[str, Any]:
            for key in list(data.keys()):
                if key not in cls.model_fields:
                    if "extra" not in data:
                        data["extra"] = {}
                    data["extra"][key] = data[key]
            return data

    __callbacks__: tuple[BaseCallback, ...] = ()
    __extra__ = None
    __requests__: tuple[Any, ...] = ()
    __response__: Any = None
    __specs__: pydantic.BaseModel = Specs

    def __init__(self, **kwargs) -> None:
        self.specs = self.__specs__(**self.default_specs(**kwargs))

    @property
    def name(self) -> str:
        return f"{self.specs.name}: ({self.specs.version})"

    @classmethod
    def from_yaml(cls, path_to_assets: pathlib.Path, path_to_yaml: pathlib.Path):
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
        if not isinstance(request, self.__requests__):
            logger.error(f"{self.specs.name}: invalid type for request.")
            raise TypeError(f"{self.specs.name}: invalid type for request.")

    def __call__(self, *args) -> Any:
        if len(args) == 0:
            logger.error(f"{self.specs.name}: missing request.")
            raise TypeError(f"{self.specs.name}: missing request.")
        if len(args) == 1:
            with timer.Timer(name=self.name):
                return self.__process__(*args)
        return [self.__process__(request=arg) for arg in args]

    def __process__(self, request: Any) -> Any:
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
        if self.specs.clear_storage and hasattr(request, "clear_storage"):
            request.clear_storage()
        if self.specs.clear_storage and hasattr(response, "clear_storage"):
            response.clear_storage()
        return response

    @abstractmethod
    def process(self, request: Any) -> Any: ...

    @abstractmethod
    def default_response(self) -> Any: ...

    def default_specs(self, **kwargs) -> dict[str, Any]:
        return kwargs

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        pos = x > 0
        neg = ~pos
        x[pos] = 1 / (1 + np.exp(-x[pos]))
        x[neg] = np.exp(x[neg]) / (1 + np.exp(x[neg]))
        return x

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        xexp = np.exp(x - x.max(-1, keepdims=True))
        return xexp / xexp.sum(-1, keepdims=True)

    def __repr__(self) -> str:
        return f"{self.specs.name}{self.specs.model_dump_json(exclude={'clear_storage'}, indent=2)}"
