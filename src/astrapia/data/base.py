import base64
from typing import Annotated, Any

import numpy as np
import pydantic


__MAGIC__: str = "@$tr@46="


class BaseData(pydantic.BaseModel, arbitrary_types_allowed=True, extra="ignore"):
    storage: Annotated[dict[str, Any], pydantic.Field(default={})]

    def clear_storage(self) -> None:
        self.storage = {}

    def model_dump(self, **kwargs) -> dict[str, Any]:
        kwargs["exclude"] = kwargs["exclude"] if "exclude" in kwargs else set()
        kwargs["exclude"].add("storage")
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        self.clear_storage()
        kwargs["exclude"] = kwargs["exclude"] if "exclude" in kwargs else set()
        kwargs["exclude"].add("storage")
        return super().model_dump_json(**kwargs)

    @staticmethod
    def encode(data: np.ndarray) -> str:
        # 'utf-8' ndarray encoding for json serialization.
        if not isinstance(data, np.ndarray):
            raise TypeError("BaseData.encode: tensor must be >= 1-dimension ndarray.")
        if data.ndim == 0:
            raise ValueError("BaseData.encode: tensor must be >= 1-dimension ndarray.")

        magic: str = f"{data.dtype!s}/{__MAGIC__}"
        if len(data.shape) >= 2:
            magic = f"{'/'.join(str(n) for n in data.shape)}/{magic}"
        return base64.b64encode(magic.encode() + data.tobytes()).decode("utf-8")

    @staticmethod
    def decode(data: str) -> np.ndarray:
        splits = base64.b64decode(data).split(__MAGIC__.encode())
        if len(splits) != 2:
            raise ValueError("BaseData.decode: data does not contain the MAGIC string.")

        *subsplits, dtype = splits[0].decode().strip()[:-1].split("/")
        shape = tuple(map(int, subsplits)) if len(subsplits) else (-1,)
        return np.frombuffer(splits[1], dtype).reshape(*shape)
