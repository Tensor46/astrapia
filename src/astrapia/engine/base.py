import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import pydantic

from astrapia import assets
from astrapia.callbacks.base import BaseCallback


logger = logging.getLogger("BaseProcess")


class BaseProcess(ABC):
    """Base process."""

    class Specs(pydantic.BaseModel):
        name: str
        version: pydantic.constr(to_lower=True)

    __callbacks__: tuple[BaseCallback, ...] = ()
    __extra__ = None
    __requests__: tuple[Any, ...] = ()
    __response__: Any = None
    __specs__: pydantic.BaseModel = Specs

    def __init__(self, **kwargs) -> None:
        self.specs = self.__specs__(**kwargs)

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
            logger.error(f"{self.__class__.__name__}: invalid type for request.")
            raise TypeError(f"{self.__class__.__name__}: invalid type for request.")

    def __call__(self, *args) -> dict[str, Any]:
        if len(args) == 0:
            logger.error(f"{self.__class__.__name__}: missing request.")
            raise TypeError(f"{self.__class__.__name__}: missing request.")
        if len(args) == 1:
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
        return response

    @abstractmethod
    def process(self, request: Any) -> Any: ...

    @abstractmethod
    def default_response(self) -> Any: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.specs.model_dump_json(indent=2)}"
