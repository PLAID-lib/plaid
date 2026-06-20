"""Generic JSON value codec with fast array encoding.

This module serializes arbitrary JSON-shaped values to a language-neutral
payload. NumPy arrays are encoded as base64 little-endian C-contiguous bytes
with explicit dtype and shape metadata, so the payload can be decoded from
Python, MATLAB, R, JavaScript, or any language with base64 and typed-array
support. Scalars pass through unchanged, and lists and dictionaries are
encoded element-wise.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

ARRAY_ENCODING = "base64"
BYTE_ORDER = "little"

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def encode_json_value(value: Any) -> Any:
    """Recursively encode a value into a JSON-compatible structure.

    Scalars pass through unchanged. Lists and tuples are encoded element-wise.
    Dictionaries are encoded value-wise with string keys. NumPy scalars,
    NumPy arrays, and bytes are encoded with the portable base64 schema.

    Args:
        value: Value to encode.

    Returns:
        A JSON-compatible representation of ``value``.

    Raises:
        TypeError: If a leaf value type is not supported.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        return {str(key): encode_json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [encode_json_value(item) for item in value]

    return encode_leaf_value(value)


def decode_json_value(value: Any) -> Any:
    """Recursively decode a value produced by :func:`encode_json_value`.

    Encoded NumPy arrays and bytes are detected by their ``kind`` marker and
    restored. Other dictionaries and lists are decoded element-wise, and
    scalars pass through unchanged.

    Args:
        value: Value to decode.

    Returns:
        The decoded value.
    """
    if isinstance(value, dict):
        if value.get("kind") in ("ndarray", "bytes"):
            return decode_leaf_value(value)
        return {key: decode_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [decode_json_value(item) for item in value]

    return value


def encode_leaf_value(value: Any) -> Any:
    """Encode a single leaf value (scalar, NumPy scalar/array, or bytes).

    Args:
        value: Leaf value to encode.

    Returns:
        A JSON-compatible representation of ``value``.

    Raises:
        TypeError: If the value type is not supported.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "encoding": ARRAY_ENCODING,
            "data": base64.b64encode(value).decode("ascii"),
        }

    if isinstance(value, np.generic):
        return encode_leaf_value(value.item())

    if isinstance(value, np.ndarray):
        return _encode_array(value)

    if isinstance(value, (list, tuple)):
        return [encode_leaf_value(item) for item in value]

    raise TypeError(f"Unsupported value type for JSON serialization: {type(value)!r}")


def decode_leaf_value(value: Any) -> Any:
    """Decode a single leaf value produced by :func:`encode_leaf_value`.

    Args:
        value: Leaf value to decode.

    Returns:
        The decoded value.
    """
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "ndarray":
            return _decode_array(value)
        if kind == "bytes":
            return base64.b64decode(value["data"])
    if isinstance(value, list):
        return [decode_leaf_value(item) for item in value]
    return value


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    """Encode a NumPy array using portable metadata plus base64 bytes."""
    array = np.asarray(value)

    if array.dtype.kind == "O":
        raise TypeError("Object dtype arrays are not supported in JSON payloads")

    if array.dtype.kind == "U":
        return {
            "kind": "ndarray",
            "encoding": "json",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "order": "C",
            "byte_order": "not-applicable",
            "data": array.tolist(),
        }

    contiguous = np.ascontiguousarray(array)
    byte_order = "not-applicable"
    if contiguous.dtype.byteorder not in ("|", "=") or contiguous.dtype.itemsize > 1:
        contiguous = contiguous.astype(contiguous.dtype.newbyteorder("<"), copy=False)
        byte_order = BYTE_ORDER

    return {
        "kind": "ndarray",
        "encoding": ARRAY_ENCODING,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "order": "C",
        "byte_order": byte_order,
        "data": base64.b64encode(contiguous.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(value: dict[str, Any]) -> np.ndarray:
    """Decode an array object from the JSON payload schema."""
    encoding = value.get("encoding")
    dtype = np.dtype(value["dtype"])
    shape = tuple(value["shape"])

    if encoding == "json":
        return np.array(value["data"], dtype=dtype).reshape(shape)

    if encoding != ARRAY_ENCODING:
        raise ValueError(f"Unsupported ndarray encoding: {encoding!r}")

    raw = base64.b64decode(value["data"])
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
</content>