import abc
from typing import Any


class BaseCallback(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None: ...

    def before_process(self, request: Any) -> Any:
        return request

    def after_process(self, response: Any) -> Any:
        return response
