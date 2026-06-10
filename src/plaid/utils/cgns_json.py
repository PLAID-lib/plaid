"""JSON serialization helpers for CGNS trees.

The helpers in this module serialize a single pyCGNS-style tree node of the
form ``[name, value, children, label]`` to a language-neutral JSON payload.
NumPy arrays are encoded as base64 little-endian C-contiguous bytes with
explicit dtype and shape metadata so the payload can be decoded from Python,
MATLAB, R, JavaScript, or any language with base64 and typed-array support.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

FORMAT_NAME = "plaid-cgns-tree-json"
FORMAT_VERSION = 1
ARRAY_ENCODING = "base64"
BYTE_ORDER = "little"

JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def cgns_tree_to_json_payload(tree: list[Any]) -> dict[str, Any]:
    """Convert a CGNS tree to a JSON-compatible payload.

    Args:
        tree: pyCGNS-style node ``[name, value, children, label]``.

    Returns:
        A JSON-compatible dictionary containing format metadata and the encoded
        tree.
    """
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "array_encoding": ARRAY_ENCODING,
        "byte_order": BYTE_ORDER,
        "tree": _encode_node(tree),
    }


def cgns_tree_from_json_payload(payload: dict[str, Any]) -> list[Any]:
    """Rebuild a CGNS tree from a JSON-compatible payload.

    Args:
        payload: Payload produced by :func:`cgns_tree_to_json_payload`.

    Returns:
        A pyCGNS-style node ``[name, value, children, label]``.

    Raises:
        ValueError: If the payload format or version is unsupported.
    """
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(f"Unsupported CGNS JSON format: {payload.get('format')!r}")
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported CGNS JSON version: {payload.get('version')!r}")
    return _decode_node(payload["tree"])


def cgns_tree_to_json(tree: list[Any], **json_kwargs: Any) -> str:
    """Convert a CGNS tree to a JSON string.

    Args:
        tree: pyCGNS-style node ``[name, value, children, label]``.
        **json_kwargs: Extra keyword arguments forwarded to :func:`json.dumps`.

    Returns:
        A JSON string containing the encoded CGNS tree.
    """
    return json.dumps(cgns_tree_to_json_payload(tree), **json_kwargs)


def cgns_tree_from_json(text: str) -> list[Any]:
    """Rebuild a CGNS tree from a JSON string.

    Args:
        text: JSON string produced by :func:`cgns_tree_to_json`.

    Returns:
        A pyCGNS-style node ``[name, value, children, label]``.
    """
    payload = json.loads(text)
    return cgns_tree_from_json_payload(payload)


def _encode_node(node: list[Any]) -> dict[str, Any]:
    """Encode one pyCGNS-style node as a JSON-compatible dictionary."""
    if not isinstance(node, list) or len(node) != 4:
        raise ValueError(
            "CGNS nodes must be lists of the form [name, value, children, label]"
        )

    name, value, children, label = node
    if children is None:
        children = []
    if not isinstance(children, list):
        raise ValueError(f"Children of CGNS node {name!r} must be a list")

    return {
        "name": str(name),
        "label": str(label),
        "value": _encode_value(value),
        "children": [_encode_node(child) for child in children],
    }


def _decode_node(node: dict[str, Any]) -> list[Any]:
    """Decode one JSON node dictionary into pyCGNS-style node form."""
    if not isinstance(node, dict):
        raise ValueError("Encoded CGNS nodes must be dictionaries")
    required = {"name", "label", "value", "children"}
    missing = required - set(node)
    if missing:
        raise ValueError(f"Encoded CGNS node is missing keys: {sorted(missing)}")
    if not isinstance(node["children"], list):
        raise ValueError(
            f"Children of encoded CGNS node {node['name']!r} must be a list"
        )

    return [
        node["name"],
        _decode_value(node["value"]),
        [_decode_node(child) for child in node["children"]],
        node["label"],
    ]


def _encode_value(value: Any) -> Any:
    """Encode a CGNS node value into JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "encoding": ARRAY_ENCODING,
            "data": base64.b64encode(value).decode("ascii"),
        }

    if isinstance(value, np.generic):
        return _encode_value(value.item())

    if isinstance(value, np.ndarray):
        return _encode_array(value)

    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]

    raise TypeError(
        f"Unsupported CGNS value type for JSON serialization: {type(value)!r}"
    )


def _decode_value(value: Any) -> Any:
    """Decode one JSON-compatible CGNS value."""
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "ndarray":
            return _decode_array(value)
        if kind == "bytes":
            return base64.b64decode(value["data"])
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _encode_array(value: np.ndarray) -> dict[str, Any]:
    """Encode a NumPy array using portable metadata plus base64 bytes."""
    array = np.asarray(value)

    if array.dtype.kind == "O":
        raise TypeError("Object dtype arrays are not supported in CGNS JSON payloads")

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
