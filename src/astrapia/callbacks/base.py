import abc
from typing import Any


class BaseCallback(abc.ABC):
    """Abstract base class for defining hooks around a processing workflow.

    Subclasses must implement initialization, and can override:
        - before_process: to modify or inspect the input request.
        - after_process: to modify or inspect the output response.
    """

    @abc.abstractmethod
    def __init__(self) -> None:
        """Initialize the callback instance."""
        ...

    def before_process(self, request: Any) -> Any:
        """Hook executed before the main processing step.

        Args:
            request (Any): The incoming request object.

        Returns:
            request (Any): The (optionally modified) request to be processed.
        """
        return request

    def after_process(self, response: Any) -> Any:
        """Hook executed after the main processing step.

        Args:
            response (Any): The output from processing the request.

        Returns:
            response (Any): The (optionally modified) response to be returned.
        """
        return response
