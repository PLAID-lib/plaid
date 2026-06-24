"""Tests for the generic JSON value codec."""

import base64

import numpy as np
import pytest

from plaid.utils.json_codec import (
    ARRAY_ENCODING,
    decode_json_value,
    decode_leaf_value,
    encode_json_value,
    encode_leaf_value,
)


@pytest.mark.parametrize("value", [None, True, False, 0, 42, -1.5, "hello", ""])
def test_scalars_round_trip_unchanged(value: object) -> None:
    """Scalars pass through encode/decode unchanged."""
    encoded = encode_json_value(value)
    assert encoded == value
    assert decode_json_value(encoded) == value


def test_numpy_scalar_is_encoded_as_python_scalar() -> None:
    """NumPy scalars are encoded as their plain Python equivalents."""
    encoded = encode_json_value(np.float64(3.5))
    assert encoded == 3.5
    assert isinstance(encoded, float)


def test_ndarray_uses_base64_raw_bytes() -> None:
    """NumPy arrays are encoded as base64 of raw little-endian C bytes."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f8")
    encoded = encode_json_value(array)

    assert encoded["kind"] == "ndarray"
    assert encoded["encoding"] == ARRAY_ENCODING
    assert encoded["shape"] == [2, 2]
    assert encoded["order"] == "C"

    expected = base64.b64encode(np.ascontiguousarray(array).tobytes(order="C")).decode(
        "ascii"
    )
    assert encoded["data"] == expected


def test_ndarray_round_trip_preserves_values_and_dtype() -> None:
    """NumPy arrays round-trip with identical values, dtype, and shape."""
    array = np.arange(12, dtype=np.int32).reshape(3, 4)
    decoded = decode_json_value(encode_json_value(array))

    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == array.dtype
    assert decoded.shape == array.shape
    np.testing.assert_array_equal(decoded, array)


def test_bytes_round_trip() -> None:
    """Raw bytes round-trip through base64 encoding."""
    raw = b"\x00\x01\x02payload"
    encoded = encode_json_value(raw)

    assert encoded["kind"] == "bytes"
    decoded = decode_json_value(encoded)
    assert decoded == raw


def test_nested_dict_with_arrays_round_trips() -> None:
    """Arrays nested inside dictionaries are encoded and restored."""
    payload = {
        "Global/P": np.array([1.0, 2.0, 3.0]),
        "meta": {"name": "case", "id": 7},
    }
    decoded = decode_json_value(encode_json_value(payload))

    assert decoded["meta"] == {"name": "case", "id": 7}
    np.testing.assert_array_equal(decoded["Global/P"], np.array([1.0, 2.0, 3.0]))


def test_nested_list_with_arrays_round_trips() -> None:
    """Arrays nested inside lists are encoded and restored."""
    payload = [np.array([1, 2]), {"x": np.array([3.0])}, "tag"]
    decoded = decode_json_value(encode_json_value(payload))

    np.testing.assert_array_equal(decoded[0], np.array([1, 2]))
    np.testing.assert_array_equal(decoded[1]["x"], np.array([3.0]))
    assert decoded[2] == "tag"


def test_string_dtype_array_round_trips() -> None:
    """Unicode arrays round-trip via the JSON list encoding."""
    array = np.array(["a", "bb", "ccc"])
    decoded = decode_json_value(encode_json_value(array))

    assert isinstance(decoded, np.ndarray)
    assert decoded.tolist() == ["a", "bb", "ccc"]


def test_dict_keys_are_coerced_to_strings() -> None:
    """Non-string dictionary keys are coerced to strings on encode."""
    encoded = encode_json_value({1: "one", 2: "two"})
    assert set(encoded) == {"1", "2"}


def test_object_dtype_array_is_rejected() -> None:
    """Object-dtype arrays cannot be serialized and raise TypeError."""
    array = np.array([object()], dtype=object)
    with pytest.raises(TypeError, match="Object dtype arrays are not supported"):
        encode_json_value(array)


def test_unsupported_object_raises_type_error() -> None:
    """A plain object leaf raises TypeError on encode."""

    class _Custom:
        pass

    with pytest.raises(TypeError, match="Unsupported value type"):
        encode_json_value(_Custom())


def test_leaf_helpers_and_unknown_encoding() -> None:
    """Cover leaf-helper list branches, numpy scalars, and bad encoding."""
    # Non-float-subclass numpy scalar is encoded as a plain Python scalar.
    assert encode_leaf_value(np.int32(7)) == 7
    # encode_leaf_value on a list encodes element-wise.
    assert encode_leaf_value([np.int32(1), 2]) == [1, 2]
    # decode_leaf_value on a list decodes element-wise.
    assert decode_leaf_value([{"kind": "bytes", "data": "AA=="}]) == [b"\x00"]
    # Unknown ndarray encoding raises ValueError.
    with pytest.raises(ValueError, match="Unsupported ndarray encoding"):
        decode_json_value(
            {
                "kind": "ndarray",
                "encoding": "bogus",
                "dtype": "<i4",
                "shape": [0],
                "data": "",
            }
        )
