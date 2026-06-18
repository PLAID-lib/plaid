"""Tests for language-neutral CGNS JSON serialization helpers."""

import json

import numpy as np
import pytest

from plaid.utils.cgns_helper import compare_cgns_trees
from plaid.utils.cgns_json import (
    _decode_array,
    _decode_node,
    _decode_value,
    _encode_node,
    _encode_value,
    cgns_tree_from_json,
    cgns_tree_from_json_payload,
    cgns_tree_to_json,
    cgns_tree_to_json_payload,
)


def _assert_no_numpy_objects(value):
    """Assert recursively that a JSON payload contains no NumPy objects."""
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_numpy_objects(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_numpy_objects(item)
    else:
        assert not isinstance(value, (np.ndarray, np.generic))


def test_cgns_tree_json_roundtrip_with_numpy_arrays():
    """A simple CGNS tree with common array dtypes survives JSON roundtrip."""
    tree = [
        "CGNSTree",
        None,
        [
            [
                "Base_2_2",
                np.array([2, 2], dtype=np.int32),
                [
                    [
                        "CoordinateX",
                        np.array([1.0, 2.0, 3.0], dtype=np.float64),
                        [],
                        "DataArray_t",
                    ],
                    [
                        "Connectivity",
                        np.array([[1, 2], [2, 3]], dtype=np.int64),
                        [],
                        "IndexArray_t",
                    ],
                    [
                        "FamilyName",
                        np.array([b"A", b"B"], dtype="|S1"),
                        [],
                        "DataArray_t",
                    ],
                    [
                        "UnicodeName",
                        np.array(["alpha", "beta"], dtype="<U5"),
                        [],
                        "DataArray_t",
                    ],
                ],
                "CGNSBase_t",
            ]
        ],
        "CGNSTree_t",
    ]

    payload = cgns_tree_to_json_payload(tree)
    _assert_no_numpy_objects(payload)
    text = json.dumps(payload)

    decoded = cgns_tree_from_json(text)

    assert compare_cgns_trees(tree, decoded)
    assert decoded[2][0][1].dtype == np.dtype("<i4")
    assert decoded[2][0][2][0][1].dtype == np.dtype("<f8")
    assert decoded[2][0][2][1][1].dtype == np.dtype("<i8")
    assert decoded[2][0][2][2][1].dtype == np.dtype("|S1")
    assert decoded[2][0][2][3][1].dtype == np.dtype("<U5")


def test_real_sample_cgns_tree_json_roundtrip(sample_with_tree):
    """A real Sample CGNS tree can be sent through JSON and reconstructed."""
    tree = sample_with_tree.data[0.0]

    text = cgns_tree_to_json(tree)
    decoded = cgns_tree_from_json(text)

    assert compare_cgns_trees(tree, decoded)


def test_cgns_tree_json_payload_roundtrip_with_json_scalars():
    """JSON scalar node values remain valid JSON scalar values."""
    tree = [
        "Root",
        None,
        [
            ["String", "value", [], "UserDefinedData_t"],
            ["Integer", 3, [], "UserDefinedData_t"],
            ["Float", 1.5, [], "UserDefinedData_t"],
            ["Boolean", True, [], "UserDefinedData_t"],
        ],
        "CGNSTree_t",
    ]

    decoded = cgns_tree_from_json_payload(cgns_tree_to_json_payload(tree))

    assert decoded == tree


def test_cgns_tree_json_rejects_invalid_payloads():
    """Invalid schema metadata and malformed nodes raise explicit errors."""
    with pytest.raises(ValueError, match="Unsupported CGNS JSON format"):
        cgns_tree_from_json_payload({"format": "other", "version": 1, "tree": {}})

    with pytest.raises(ValueError, match="Unsupported CGNS JSON version"):
        cgns_tree_from_json_payload(
            {"format": "plaid-cgns-tree-json", "version": 999, "tree": {}}
        )

    with pytest.raises(ValueError, match="CGNS nodes must be lists"):
        cgns_tree_to_json_payload(["Root", None, [], "CGNSTree_t", "extra"])


def test_cgns_tree_json_rejects_object_arrays():
    """Object dtype arrays are intentionally not part of the portable schema."""
    tree = [
        "Root",
        np.array([{"not": "portable"}], dtype=object),
        [],
        "CGNSTree_t",
    ]

    with pytest.raises(TypeError, match="Object dtype arrays"):
        cgns_tree_to_json_payload(tree)


def test_encode_node_accepts_none_children_as_empty_list():
    """CGNS nodes with None children are normalized to an empty child list."""
    encoded = _encode_node(["Root", None, None, "CGNSTree_t"])

    assert encoded == {
        "name": "Root",
        "label": "CGNSTree_t",
        "value": None,
        "children": [],
    }


@pytest.mark.parametrize(
    "node, message",
    [
        (("Root", None, [], "CGNSTree_t"), "CGNS nodes must be lists"),
        (["Root", None, [], "CGNSTree_t", "extra"], "CGNS nodes must be lists"),
        (["Root", None, "not-a-list", "CGNSTree_t"], "Children of CGNS node"),
    ],
)
def test_encode_node_rejects_malformed_nodes(node, message):
    """Malformed pyCGNS-style nodes raise explicit errors before encoding."""
    with pytest.raises(ValueError, match=message):
        _encode_node(node)


@pytest.mark.parametrize(
    "encoded, message",
    [
        ([], "Encoded CGNS nodes must be dictionaries"),
        ({"name": "Root", "label": "CGNSTree_t", "value": None}, "missing keys"),
        (
            {
                "name": "Root",
                "label": "CGNSTree_t",
                "value": None,
                "children": "not-a-list",
            },
            "Children of encoded CGNS node",
        ),
    ],
)
def test_decode_node_rejects_malformed_encoded_nodes(encoded, message):
    """Malformed encoded node dictionaries are rejected before decoding."""
    with pytest.raises(ValueError, match=message):
        _decode_node(encoded)


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int32(7), 7),
        (np.float64(1.25), 1.25),
        ((np.int64(1), np.int64(2)), [1, 2]),
    ],
)
def test_encode_value_normalizes_numpy_scalars_and_tuples(value, expected):
    """NumPy scalar values and tuples are converted to JSON-compatible data."""
    assert _encode_value(value) == expected


def test_encode_decode_value_roundtrips_bytes():
    """Bytes values are encoded as base64 objects and decoded back to bytes."""
    value = b"CGNS bytes"

    encoded = _encode_value(value)

    assert encoded["kind"] == "bytes"
    assert _decode_value(encoded) == value


def test_decode_value_decodes_nested_lists():
    """List payloads are decoded recursively."""
    encoded_bytes = _encode_value(b"nested bytes")

    assert _decode_value([encoded_bytes, [1, encoded_bytes]]) == [
        b"nested bytes",
        [1, b"nested bytes"],
    ]


def test_encode_value_rejects_unsupported_values():
    """Unsupported value types raise a TypeError with a clear message."""
    with pytest.raises(TypeError, match="Unsupported CGNS value type"):
        _encode_value({"not": "a supported CGNS value"})


def test_decode_value_leaves_unknown_dict_kind_unchanged():
    """Unknown dictionary payloads are passed through unchanged."""
    value = {"kind": "custom", "data": [1, 2, 3]}

    assert _decode_value(value) is value


def test_decode_array_rejects_unknown_encoding():
    """Only JSON and base64 ndarray encodings are supported."""
    with pytest.raises(ValueError, match="Unsupported ndarray encoding"):
        _decode_array(
            {
                "encoding": "unsupported",
                "dtype": "<f8",
                "shape": [0],
                "data": "",
            }
        )
