"""Tests for common storage preprocessing helpers."""

import numpy as np

from plaid.storage.common.preprocessor import infer_dtype


def test_infer_dtype_detects_single_byte_string_arrays():
    """Byte-array encoded CGNS strings should use the canonical S1 dtype."""
    value = np.array([b"A", b"B"], dtype="S1")

    assert infer_dtype(value) == {"dtype": "S1", "ndim": 1}
