"""JSON serialization helpers for CGNS trees.

The helpers in this module serialize a single pyCGNS-style tree node of the
form ``[name, value, children, label]`` to a language-neutral JSON payload.
Node values are encoded with :mod:`plaid.utils.json_codec`, which stores NumPy
arrays as base64 little-endian C-contiguous bytes with explicit dtype and shape
metadata so the payload can be decoded from Python, MATLAB, R, JavaScript, or
any language with base64 and typed-array support.
"""

from __future__ import annotations

import json
from typing import Any

from .json_codec import decode_leaf_value, encode_leaf_value

FORMAT_NAME = "plaid-cgns-tree-json"
FORMAT_VERSION = 1


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
        "value": encode_leaf_value(value),
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
        decode_leaf_value(node["value"]),
        [_decode_node(child) for child in node["children"]],
        node["label"],
    ]
