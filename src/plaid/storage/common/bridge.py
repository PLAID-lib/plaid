# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Common bridge utilities.

This module provides bridge functions for converting between PLAID samples and
storage formats, including flattening/unflattening and sample reconstruction.
"""

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from typing import Any, Optional

import numpy as np

from plaid import Sample
from plaid.containers.features import SampleFeatures
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.storage.common.tree_handling import unflatten_cgns_tree


def unflatten_path(key: str) -> str:
    """Unflattens a Zarr key by replacing underscores with slashes.

    Args:
        key (str): The flattened key with underscores.

    Returns:
        str: The unflattened key with slashes.
    """
    return key.replace("__", "/")


def flatten_path(key: str) -> str:
    """Flattens a path key by replacing slashes with underscores.

    Args:
        key (str): The path key to flatten.

    Returns:
        str: The flattened key with slashes replaced by underscores.
    """
    return key.replace("/", "__")


def _split_dict(d: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a dictionary into values and times based on key suffixes.

    Args:
        d: Dictionary with keys that may end with '_times'.

    Returns:
        tuple: (vals, times) where vals has non-times keys, times has times keys.
    """
    vals = {}
    times = {}
    for k, v in d.items():
        if k.endswith("_times"):
            times[k[:-6]] = v
        else:
            vals[k] = v
    return vals, times


def _split_dict_feat(
    d: dict[str, Any], features_set: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover
    """Split a dictionary into values and times, filtering by features set.

    Args:
        d: Dictionary with keys.
        features_set: Set of feature names to include.

    Returns:
        tuple: (vals, times) filtered by features_set.
    """
    vals = {}
    times = {}
    for k, v in d.items():
        if k.endswith("_times") and k[:-6] in features_set:
            times[k[:-6]] = v
        elif k in features_set:
            vals[k] = v
    return vals, times


def to_sample_dict(
    var_sample_dict: dict[str, Any],
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
    features: Optional[list[str]] = None,
) -> dict[float, dict[str, Any]]:
    """Convert variable sample dict to time-based sample dict.

    Args:
        var_sample_dict: Variable features dictionary.
        flat_cst: Constant features dictionary.
        cgns_types: CGNS types dictionary.
        features: Optional list of features to include.

    Returns:
        dict: Time-based sample dictionary.
    """
    assert not isinstance(flat_cst[next(iter(flat_cst))], dict), (
        "did you provide the complete `flat_cst` instead of the one for the considered split?"
    )

    if features is None:
        flat_cst_val, flat_cst_tim = _split_dict(flat_cst)
        row_val, row_tim = _split_dict(var_sample_dict)
    else:  # pragma: no cover
        features_set = set(features)
        flat_cst_val, flat_cst_tim = _split_dict_feat(flat_cst, features_set)
        row_val, row_tim = _split_dict_feat(var_sample_dict, features_set)

    row_val.update(flat_cst_val)
    row_tim.update(flat_cst_tim)

    row_val = {p: row_val[p] for p in sorted(row_val)}
    row_tim = {p: row_tim[p] for p in sorted(row_tim)}

    print("row_val =", row_val)
    print("row_tim =", row_tim)

    sample_flat_trees = {}
    paths_none = {}
    for (path_t, times_struc), (path_v, val) in zip(row_tim.items(), row_val.items()):
        print(">>", path_t, path_v)
        assert path_t == path_v, "did you forget to specify the features arg?"
        if val is None:
            assert times_struc is None
            if path_v not in paths_none and cgns_types[path_v] not in [
                "DataArray_t",
                "IndexArray_t",
            ]:
                paths_none[path_v] = None
        else:
            times_struc = np.array(times_struc, dtype=np.float64).reshape((-1, 3))
            for i, time in enumerate(times_struc[:, 0]):
                start = int(times_struc[i, 1])
                end = int(times_struc[i, 2])
                if end == -1:
                    end = None
                if val.ndim > 1:
                    values = val[:, start:end]
                else:
                    values = val[start:end]
                    if isinstance(values[0], str):
                        values = np.frombuffer(
                            values[0].encode("ascii", "strict"), dtype="|S1"
                        )
                if time in sample_flat_trees:
                    sample_flat_trees[time][path_v] = values
                else:
                    sample_flat_trees[time] = {path_v: values}

    def _wants(path: str) -> bool:
        return features is None or path in features

    for time, tree in sample_flat_trees.items():
        bases = {k.split("/", 1)[0] for k in tree}

        for base in bases:
            time_base = f"{base}/Time"

            if _wants(time_base):
                tree[time_base] = np.array([1], dtype=np.int32)

            if _wants(f"{time_base}/IterationValues"):
                tree[f"{time_base}/IterationValues"] = np.array([1], dtype=np.int32)

            if _wants(f"{time_base}/TimeValues"):
                tree[f"{time_base}/TimeValues"] = np.array([time], dtype=np.float64)

        if _wants("CGNSLibraryVersion"):
            tree["CGNSLibraryVersion"] = np.array([4.0], dtype=np.float32)

        tree.update(paths_none)

    print("sample_flat_trees =", sample_flat_trees)

    return sample_flat_trees


def to_sample_dict_2(
    sample_dict: dict[str, Any],
    cgns_types: dict[str, str],
    features: Optional[list[str]] = None,
) -> dict[float, dict[str, Any]]:
    """Convert a flattened sample dict into a time-indexed flat-tree mapping.

    This function processes a single flattened sample dictionary where keys
    are feature paths or feature path suffixed with '_times' (holding time
    interval specs). It splits values and their corresponding time
    specifications, slices feature arrays per interval and groups features
    by their sample time into CGNS-style flattened trees.

    Args:
        sample_dict: Mapping of flattened feature paths to values or to time
            specifications (keys ending with '_times'). Time specifications
            must be sequences or arrays reshapeable to (-1, 3) rows of
            (time, start, end).
        cgns_types: Mapping from feature path to CGNS node type; used to
            determine which None-valued features should be preserved.
        features: Optional list of feature paths to include; if None all
            features are included.

    Returns:
        A dict mapping time (float) to a flattened tree dict (path -> array or None).
    """
    if features is None:
        row_val, row_tim = _split_dict(sample_dict)
    else:  # pragma: no cover
        features_set = set(features)
        row_val, row_tim = _split_dict_feat(sample_dict, features_set)

    row_val = {p: row_val[p] for p in sorted(row_val)}
    row_tim = {p: row_tim[p] for p in sorted(row_tim)}

    sample_flat_trees = {}
    paths_none = {}
    for (path_t, times_struc), (path_v, val) in zip(row_tim.items(), row_val.items()):
        assert path_t == path_v, "did you forget to specify the features arg?"
        if val is None:
            assert times_struc is None
            if path_v not in paths_none and cgns_types[path_v] not in [
                "DataArray_t",
                "IndexArray_t",
            ]:
                paths_none[path_v] = None
        else:
            times_struc = np.array(times_struc, dtype=np.float64).reshape((-1, 3))
            for i, time in enumerate(times_struc[:, 0]):
                start = int(times_struc[i, 1])
                end = int(times_struc[i, 2])
                if end == -1:
                    end = None
                if val.ndim > 1:
                    values = val[:, start:end]
                else:
                    values = val[start:end]
                    if isinstance(values[0], str):
                        values = np.frombuffer(
                            values[0].encode("ascii", "strict"), dtype="|S1"
                        )
                if time in sample_flat_trees:
                    sample_flat_trees[time][path_v] = values
                else:
                    sample_flat_trees[time] = {path_v: values}

    def _wants(path: str) -> bool:
        return features is None or path in features

    for time, tree in sample_flat_trees.items():
        bases = {k.split("/", 1)[0] for k in tree}

        for base in bases:
            time_base = f"{base}/Time"

            if _wants(time_base):
                tree[time_base] = np.array([1], dtype=np.int32)

            if _wants(f"{time_base}/IterationValues"):
                tree[f"{time_base}/IterationValues"] = np.array([1], dtype=np.int32)

            if _wants(f"{time_base}/TimeValues"):
                tree[f"{time_base}/TimeValues"] = np.array([time], dtype=np.float64)

        if _wants("CGNSLibraryVersion"):
            tree["CGNSLibraryVersion"] = np.array([4.0], dtype=np.float32)

        tree.update(paths_none)

    return sample_flat_trees


def to_plaid_sample(
    sample_dict: dict[float, dict[str, Any]],
    cgns_types: dict[str, str],
) -> Sample:
    """Convert sample dict to PLAID Sample.

    Args:
        sample_dict: Time-based sample dictionary.
        cgns_types: CGNS types dictionary.

    Returns:
        Sample: The reconstructed PLAID Sample.
    """
    sample_data = {}
    for time, flat_tree in sample_dict.items():
        sample_data[time] = unflatten_cgns_tree(flat_tree, cgns_types)

    return Sample(path=None, features=SampleFeatures(sample_data))


def plaid_to_sample_dict(
    sample: Sample, variable_schema: dict[str, Any], constant_schema: dict[str, Any]
) -> dict[str, Any]:
    """Convert PLAID Sample to sample dict.

    Args:
        sample: The PLAID Sample.
        variable_schema: Variable schema dictionary.
        constant_schema: Constant schema dictionary.

    Returns:
        dict[str, Any]: sample_dict
    """
    var_features = list(variable_schema.keys())
    cst_features = list(constant_schema.keys())

    sample_dict, _, _ = build_sample_dict(sample)

    var_sample_dict = {path: sample_dict.get(path, None) for path in var_features}
    cst_sample_dict = {path: sample_dict.get(path, None) for path in cst_features}

    return cst_sample_dict | var_sample_dict
