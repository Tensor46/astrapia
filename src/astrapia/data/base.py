import base64
from typing import Annotated, Any

import numpy as np
import pydantic


# Separator token used to delimit metadata in the base64 payload.
__MAGIC__: str = "@$tr@46="


class BaseData(pydantic.BaseModel, arbitrary_types_allowed=True, extra="ignore"):
    """Base model for JSON-serializable data with auxiliary storage.

    Provides methods to clear transient storage, customize serialization
    (excluding storage), and encode/decode numpy arrays to/from base64 strings.
    """

    storage: Annotated[dict[str, Any], pydantic.Field(default={})]

    def clear_storage(self) -> None:
        """Clear all entries in the auxiliary storage."""
        self.storage = {}

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serialize model to a dict, excluding the `storage` field."""
        kwargs["exclude"] = kwargs["exclude"] if "exclude" in kwargs else set()
        kwargs["exclude"].add("storage")
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        """Serialize model to a JSON string, clearing and excluding `storage`."""
        self.clear_storage()
        kwargs["exclude"] = kwargs["exclude"] if "exclude" in kwargs else set()
        kwargs["exclude"].add("storage")
        return super().model_dump_json(**kwargs)

    @staticmethod
    def encode(data: np.ndarray) -> str:
        """Encode a 1D+ numpy array into a base64 string with dtype and shape metadata.
        The format prepends "<dim…>/<dtype>/<__MAGIC__>" before the raw bytes.

        Args:
            data (np.ndarray): Array of at least one dimension.

        Returns:
            str: Base64-encoded string with embedded metadata.

        Raises:
            TypeError: If `data` is not a numpy.ndarray.
            ValueError: If `data` is zero-dimensional.
        """
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
        """Decode a base64 string produced by `encode` back into a numpy array.

        Expects format "<shape>/<dtype>/<__MAGIC__><raw_bytes>" before base64.

        Args:
            data (str): Base64 string with embedded metadata.

        Returns:
            np.ndarray: Reconstructed array.

        Raises:
            ValueError: If the MAGIC token is missing or metadata is malformed.
        """
        splits = base64.b64decode(data).split(__MAGIC__.encode())
        if len(splits) != 2:
            raise ValueError("BaseData.decode: data does not contain the MAGIC token.")

        # remove trailing slash, then split "<dim...>/<dtype>"
        *subsplits, dtype = splits[0].decode().strip()[:-1].split("/")
        shape = tuple(map(int, subsplits)) if len(subsplits) else (-1,)
        return np.frombuffer(splits[1], dtype).reshape(*shape)
