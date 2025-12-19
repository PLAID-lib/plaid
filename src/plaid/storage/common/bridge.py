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

from typing import Any, Optional

import numpy as np

from plaid import Sample
from plaid.containers.features import SampleFeatures
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.storage.common.tree_handling import unflatten_cgns_tree


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

    for time, tree in sample_flat_trees.items():
        bases = list(set([k.split("/")[0] for k in tree.keys()]))
        for base in bases:
            tree[f"{base}/Time"] = np.array([1], dtype=np.int32)
            tree[f"{base}/Time/IterationValues"] = np.array([1], dtype=np.int32)
            tree[f"{base}/Time/TimeValues"] = np.array([time], dtype=np.float64)
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

    hf_sample, _, _ = build_sample_dict(sample)

    var_sample_dict = {path: hf_sample.get(path, None) for path in var_features}
    cst_sample_dict = {path: hf_sample.get(path, None) for path in cst_features}

    return cst_sample_dict | var_sample_dict
