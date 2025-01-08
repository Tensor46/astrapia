__all__ = ["Timer", "timer"]

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


ENABLE_TIMER: bool = False
logger = logging.getLogger("Timer")


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        total = time.time() - self.start
        if ENABLE_TIMER:
            logger.info(f"{self.name} took {total:2.4f} seconds.")
            print(f"{self.name} took {total:2.4f} seconds.")  # noqa: T201


def timer(fn: Callable, name: str) -> Callable:
    @wraps(fn)
    def fn_wrap(*args, **kwargs) -> Any:
        with Timer(name=name):
            result = fn(*args, **kwargs)
        return result

    return fn_wrap
