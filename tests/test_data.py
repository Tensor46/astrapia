import numpy as np

import astrapia


def test_tensor_encoding_decoding():
    r"""Test astrapia.data.Tensor."""
    dtypes_int = ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
    dtypes_float = ("float16", "float32", "float64")
    for dtype in dtypes_int + dtypes_float:
        for ndim in range(1, 8):
            shape = np.random.randint(0, 10, ndim).tolist()
            if "int" in dtype:
                data = np.random.randint(0, 100, shape).astype(dtype)
            else:
                data = np.random.randn(*shape).astype(dtype)

            assert np.allclose(data, astrapia.data.BaseTensor.decode(astrapia.data.BaseTensor.encode(data)))
