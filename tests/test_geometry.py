import numpy as np

import astrapia
import astrapia.geometry


def test_similarity_transform():
    r"""Test geometry.transform.similarity."""
    tmat = astrapia.geometry.transform.similarity([[4, 6], [6, 6]], [[2, 3], [3, 3]])
    assert np.allclose(tmat, [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
