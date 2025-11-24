from typing import Any, Optional

from plaid import Sample
from plaid.containers.features import SampleFeatures
from plaid.storage.common.tree_handling import unflatten_cgns_tree

import numpy as np


def _split_dict(d):
    vals = {}
    times = {}
    for k, v in d.items():
        if k.endswith("_times"):
            times[k[:-6]] = v
        else:
            vals[k] = v
    return vals, times


def _split_dict_feat(d, features_set):
    vals = {}
    times = {}
    for k, v in d.items():
        if k.endswith("_times") and k[:-6] in features_set:
            times[k[:-6]] = v
        elif k in features_set:
            vals[k] = v
    return vals, times


def to_plaid_sample(
    var_sample_dict:dict,
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
    features: Optional[list[str]] = None
) -> Sample:
    """Convert a Hugging Face dataset row to a PLAID Sample object.

    Extracts a single row from a Hugging Face dataset and converts it
    into a PLAID Sample by unflattening the CGNS tree structure. Constant features
    from flat_cst are merged with the variable features from the row.

    Args:
        ds (datasets.Dataset): The Hugging Face dataset containing the sample data.
        i (int): The index of the row to convert.
        flat_cst (dict[str, Any]): Dictionary of constant features to add to each sample.
        cgns_types (dict[str, str]): Dictionary mapping paths to CGNS types for reconstruction.
        enforce_shapes (bool, optional): If True, ensures consistent array shapes during conversion. Defaults to True.

    Returns:
        Sample: A validated PLAID Sample object reconstructed from the Hugging Face dataset row.

    Note:
        - Uses the dataset's pyarrow table data for efficient access.
        - Handles array shapes and types according to enforce_shapes.
        - Constant features from flat_cst are merged with the variable features from the row.
    """
    assert not isinstance(flat_cst[next(iter(flat_cst))], dict), (
        "did you provide the complete `flat_cst` instead of the one for the considered split?"
    )

    if features is None:
        flat_cst_val, flat_cst_times = _split_dict(flat_cst)
        row_val, row_tim = _split_dict(var_sample_dict)

    else:
        features_set = set(features)

        flat_cst_val, flat_cst_times = _split_dict_feat(flat_cst, features_set)
        row_val, row_tim = _split_dict_feat(var_sample_dict, features_set)

    row_val.update(flat_cst_val)
    row_tim.update(flat_cst_times)

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
            times_struc = times_struc.reshape((-1, 3))
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

    sample_data = {}
    for time, flat_tree in sample_flat_trees.items():
        flat_tree.update(paths_none)
        sample_data[time] = unflatten_cgns_tree(flat_tree, cgns_types)

    return Sample(path=None, features=SampleFeatures(sample_data))