"""Timing utilities: context manager and decorator for measuring code execution duration.

Toggle logging output via the `ENABLE_TIMER` flag.
"""

__all__ = ["Timer", "timer"]

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


ENABLE_TIMER: bool = False
logger = logging.getLogger("Timer")


class Timer:
    """Context manager that records and reports the elapsed time of a code block.

    Usage:
        with Timer("my task"):
            do_something()

    If ENABLE_TIMER is True, logs and prints the duration.
    """

    def __init__(self, name: str):
        """
        Args:
            name (str): Descriptive name for the timed block.
        """
        self.name = name

    def __enter__(self):
        """Start the timer.

        Returns:
            Timer: The timer instance (with `start` timestamp set).
        """
        self.start = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        """Stop the timer and report the elapsed time if enabled.

        Args:
            exc_type, exc_val, exc_tb: Exception info (ignored here).
        """
        total = time.time() - self.start
        if ENABLE_TIMER:
            logger.info(f"{self.name} took {total:2.4f} seconds.")
            print(f"{self.name} took {total:2.4f} seconds.")  # noqa: T201


def timer(fn: Callable, name: str) -> Callable:
    """Decorator to measure and report the execution time of a function.

    Wraps the function in a Timer context with the given name.

    Args:
        fn (Callable): The function to wrap.
        name (str): Descriptive name for timing output.

    Returns:
        Callable: Wrapped function that times its execution.
    """

    @wraps(fn)
    def fn_wrap(*args, **kwargs) -> Any:
        with Timer(name=name):
            result = fn(*args, **kwargs)
        return result

    return fn_wrap
