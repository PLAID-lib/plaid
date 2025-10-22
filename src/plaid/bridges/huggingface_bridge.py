"""Hugging Face bridge for PLAID datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
import hashlib
import io
import json
import os
import pickle
import shutil
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyarrow as pa
import yaml
from tqdm import tqdm

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import logging
from typing import Any, Union

import datasets
from datasets import Features, Sequence, Value, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pydantic import ValidationError

from plaid import Dataset, ProblemDefinition, Sample
from plaid.containers.features import SampleFeatures
from plaid.types import IndexType
from plaid.utils.cgns_helper import (
    flatten_cgns_tree,
    unflatten_cgns_tree,
)
from plaid.utils.deprecation import deprecated

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#     HUGGING FACE BRIDGE (with tree flattening and pyarrow tables)
# ------------------------------------------------------------------------------


def infer_hf_features_from_value(value: Any) -> Union[Value, Sequence]:
    """Infer Hugging Face dataset feature type from a given value.

    This function analyzes the input value and determines the appropriate Hugging Face
    feature type representation. It handles None values, scalars, and arrays/lists
    of various dimensions, mapping them to corresponding Hugging Face Value or Sequence types.

    Args:
        value (Any): The value to infer the feature type from. Can be None, scalar,
            list, tuple, or numpy array.

    Returns:
        datasets.Feature: A Hugging Face feature type (Value or Sequence) that corresponds
            to the input value's structure and data type.

    Raises:
        TypeError: If the value type is not supported.
        TypeError: If the array dimensionality exceeds 3D for arrays/lists.

    Notes:
        - For scalar values, maps numpy dtypes to appropriate Hugging Face Value types:
          float types to "float32", int32 to "int32", int64 to "int64", others to "string"
        - For arrays/lists, creates nested Sequence structures based on dimensionality:
          1D → Sequence(base_type), 2D → Sequence(Sequence(base_type)),
          3D → Sequence(Sequence(Sequence(base_type)))
        - All float values are enforced to "float32" to limit data size
        - All int64 values are preserved as "int64" to satisfy CGNS standards
    """
    if value is None:
        return Value("null")  # pragma: no cover

    # Scalars
    if np.isscalar(value):
        dtype = np.array(value).dtype
        if np.issubdtype(
            dtype, np.floating
        ):  # enforcing float32 for all floats, to be updated in case we want to keep float64
            return Value("float32")
        elif np.issubdtype(dtype, np.int32):
            return Value("int32")
        elif np.issubdtype(
            dtype, np.int64
        ):  # very important to satisfy the CGNS standard
            return Value("int64")
        elif np.issubdtype(dtype, np.dtype("|S1")):  # pragma: no cover
            return Value("string")
        else:
            raise ValueError("Type not recognize")  # pragma: no cover

    # Arrays / lists
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value)
        base_type = infer_hf_features_from_value(arr.flat[0] if arr.size > 0 else None)
        if arr.ndim == 1:
            return Sequence(base_type)
        elif arr.ndim == 2:
            return Sequence(Sequence(base_type))
        elif arr.ndim == 3:
            return Sequence(Sequence(Sequence(base_type)))
        else:
            raise TypeError(f"Unsupported ndim: {arr.ndim}")  # pragma: no cover
    raise TypeError(f"Unsupported type: {type(value)}")  # pragma: no cover


def build_hf_sample(sample: Sample) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """Flatten a PLAID Sample's CGNS trees into Hugging Face–compatible arrays and metadata.

    The function traverses every CGNS tree stored in sample.features.data (keyed by time),
    produces a flattened mapping path -> primitive value for each time, and then builds
    compact numpy arrays suitable for storage in a Hugging Face Dataset. Repeated value
    blocks that are identical across times are deduplicated and referenced by start/end
    indices; companion "<path>_times" arrays describe, per time, the slice indices into
    the concatenated arrays.

    Args:
        sample (Sample): A PLAID Sample whose features contain one or more CGNS trees
            (sample.features.data maps time -> CGNSTree).

    Returns:
        tuple:
            - hf_sample (dict[str, Any]): Mapping of flattened CGNS paths to either a
              numpy array (concatenation of per-time blocks) or None. For each path
              there is also an entry "<path>_times" containing a flattened numpy array
              of triplets [time, start, end] (end == -1 indicates the block extends to
              the end of the array).
            - all_paths (list[str]): Sorted list of all considered variable feature paths
              (excluding Time-related nodes and CGNSLibraryVersion).
            - sample_cgns_types (dict[str, str]): Mapping from path to CGNS node type
              (metadata produced by flatten_cgns_tree).

    Notes:
        - Byte-array encoded strings (dtype "|S1") are handled by reassembling and
          storing the string as a single-element numpy array; a sha256 hash is used
          for deduplication.
        - Deduplication reduces storage when identical blocks recur across times.
        - Paths containing "/Time" or "CGNSLibraryVersion" are ignored for variable features.
    """
    sample_flat_trees = {}
    sample_cgns_types = {}
    all_paths = set()

    # --- Flatten CGNS trees ---
    for time, tree in sample.features.data.items():
        flat, cgns_types = flatten_cgns_tree(tree)
        sample_flat_trees[time] = flat

        all_paths.update(
            k for k in flat.keys() if "/Time" not in k and "CGNSLibraryVersion" not in k
        )

        sample_cgns_types.update(cgns_types)

    hf_sample = {}

    for path in all_paths:
        hf_sample[path] = None
        hf_sample[path + "_times"] = None

        known_values = {}
        values_acc, times_acc = [], []
        current_length = 0

        for time, flat in sample_flat_trees.items():
            if path not in flat:
                continue  # pragma: no cover
            value = flat[path]

            # Handle byte-array encoded strings
            if (
                isinstance(value, np.ndarray)
                and value.dtype == np.dtype("|S1")
                and value.ndim == 1
            ):
                value_str = b"".join(value).decode("ascii")
                value_np = np.array([value_str])
                key = hashlib.sha256(value_str.encode("ascii")).hexdigest()
                size = 1
            elif value is not None:
                value_np = value
                key = hashlib.sha256(value.tobytes()).hexdigest()
                size = (
                    value.shape[-1]
                    if isinstance(value, np.ndarray) and value.ndim >= 1
                    else 1
                )
            else:
                continue

            # Deduplicate identical arrays
            if key in known_values:
                start, end = known_values[key]  # pragma: no cover
            else:
                start, end = current_length, current_length + size
                known_values[key] = (start, end)
                values_acc.append(value_np)
                current_length = end

            times_acc.append([time, start, end])

        # Build arrays
        if values_acc:
            try:
                hf_sample[path] = np.hstack(values_acc)
            except Exception:  # pragma: no cover
                hf_sample[path] = np.concatenate([np.atleast_1d(x) for x in values_acc])

            if len(known_values) == 1:
                for t in times_acc:
                    t[-1] = -1
            hf_sample[path + "_times"] = np.array(times_acc).flatten()
        else:
            hf_sample[path] = None
            hf_sample[path + "_times"] = None

    # Convert lists to numpy arrays
    for k, v in hf_sample.items():
        if isinstance(v, list):
            hf_sample[k] = np.array(v)  # pragma: no cover

    return hf_sample, all_paths, sample_cgns_types


def _generator_prepare_for_huggingface(
    generators: dict[str, Callable],
    verbose: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Features]:
    """Inspect PLAID dataset generators and infer Hugging Face feature schema.

    Iterates over all samples in all provided split generators to:
      1. Flatten each CGNS tree into a dictionary of paths → values.
      2. Infer Hugging Face `Features` types for all variable leaves.
      3. Detect constant leaves (values that never change across all samples).
      4. Collect global CGNS type metadata.

    Args:
        generators (dict[str, Callable]):
            Mapping from split names to callables returning sample generators.
            Each sample must have `sample.features.data[0.0]` compatible with `flatten_cgns_tree`.
        verbose (bool, optional): If True, displays progress bars while processing splits.

    Returns:
        tuple:
            - flat_cst (dict[str, Any]): Mapping from feature path to constant values detected across all splits.
            - key_mappings (dict[str, Any]): Metadata dictionary with:
                - "variable_features" (list[str]): paths of non-constant features.
                - "constant_features" (list[str]): paths of constant features.
                - "cgns_types" (dict[str, Any]): CGNS type information for all paths.
            - hf_features (datasets.Features): Hugging Face feature specification for variable features.

    Raises:
        ValueError: If inconsistent CGNS types or feature types are found for the same path.
    """

    def values_equal(v1, v2):
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        return v1 == v2

    global_cgns_types = {}
    global_feature_types = {}

    split_flat_cst = {}
    split_var_path = {}
    split_all_paths = {}

    # ---- Single pass over all splits and samples ----
    for split_name, generator in generators.items():
        split_constant_leaves = {}

        split_all_paths[split_name] = set()

        for sample in tqdm(
            generator(), disable=not verbose, desc=f"Process split {split_name}"
        ):
            # --- Build Hugging Face–compatible sample ---
            hf_sample, all_paths, sample_cgns_types = build_hf_sample(sample)

            split_all_paths[split_name].update(hf_sample.keys())
            # split_all_paths[split_name].update(all_paths)
            global_cgns_types.update(sample_cgns_types)

            # --- Infer global HF feature types ---
            for path in all_paths:
                value = hf_sample[path]
                if value is None:
                    continue

                if isinstance(value, np.ndarray) and value.dtype.type is np.str_:
                    inferred = Value("string")
                else:
                    inferred = infer_hf_features_from_value(value)

                if path not in global_feature_types:
                    global_feature_types[path] = inferred
                elif repr(global_feature_types[path]) != repr(inferred):
                    raise ValueError(  # pragma: no cover
                        f"Feature type mismatch for {path} in split {split_name}"
                    )

            # --- Update per-split constant detection ---
            for path, value in hf_sample.items():
                if path not in split_constant_leaves:
                    split_constant_leaves[path] = {
                        "value": value,
                        "constant": True,
                        "count": 1,
                    }
                else:
                    entry = split_constant_leaves[path]
                    entry["count"] += 1
                    if entry["constant"] and not values_equal(entry["value"], value):
                        entry["constant"] = False

        # --- Record per-split constants ---
        split_flat_cst[split_name] = dict(
            sorted(
                (
                    (p, e["value"])
                    for p, e in split_constant_leaves.items()
                    if e["constant"]
                ),
                key=lambda x: x[0],
            )
        )

        split_var_path[split_name] = {
            p
            for p in split_all_paths[split_name]
            if p not in split_flat_cst[split_name]
        }

    global_feature_types = {
        p: global_feature_types[p] for p in sorted(global_feature_types)
    }
    var_features = sorted(list(set().union(*split_var_path.values())))

    if len(var_features) == 0:
        raise ValueError(  # pragma: no cover
            "no variable feature found, is your dataset variable through samples?"
        )

    # ---------------------------------------------------
    # for test-like splits, some var_features are all None (e.g.: outputs): need to add '_times' counterparts to corresponding constant trees
    for split_name in split_flat_cst.keys():
        for path in var_features:
            if not path.endswith("_times") and path not in split_all_paths[split_name]:
                split_flat_cst[split_name][path + "_times"] = None  # pragma: no cover
            if (
                path in split_flat_cst[split_name]
            ):  # remove for flat_cst the path that will be forcely included in the arrow tables
                split_flat_cst[split_name].pop(path)  # pragma: no cover

    # ---- Constant features sanity check
    cst_features = {
        split_name: sorted(list(cst.keys()))
        for split_name, cst in split_flat_cst.items()
    }

    first_split, first_value = next(iter(cst_features.items()), (None, None))
    for split, value in cst_features.items():
        assert value == first_value, (
            f"cst_features differ for split '{split}' (vs '{first_split}'): something went wrong in _generator_prepare_for_huggingface."
        )

    cst_features = first_value

    # ---- Build global HF Features (only variable) ----
    hf_features_map = {}
    for k in var_features:
        if k.endswith("_times"):
            hf_features_map[k] = Sequence(Value("float64"))  # pragma: no cover
        else:
            hf_features_map[k] = global_feature_types[k]

    hf_features = Features(hf_features_map)

    var_features = [path for path in var_features if not path.endswith("_times")]
    cst_features = [path for path in cst_features if not path.endswith("_times")]

    key_mappings = {
        "variable_features": var_features,
        "constant_features": cst_features,
        "cgns_types": global_cgns_types,
    }

    return split_flat_cst, key_mappings, hf_features


def to_plaid_dataset(
    hf_dataset: datasets.Dataset,
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
    enforce_shapes: bool = True,
) -> Dataset:
    """Convert a Hugging Face dataset into a PLAID dataset.

    Iterates over all samples in a Hugging Face `Dataset` and converts each one
    into a PLAID-compatible sample using `to_plaid_sample`. The resulting
    samples are then collected into a single PLAID `Dataset`.

    Args:
        hf_dataset (datasets.Dataset): The Hugging Face dataset split to convert.
        flat_cst (dict[str, Any]): Flattened representation of the CGNS tree structure constants.
        cgns_types (dict[str, str]): Mapping of CGNS paths to their expected types.
        enforce_shapes (bool, optional): If True, ensures all arrays strictly follow the reference shapes. Defaults to True.

    Returns:
        Dataset: A PLAID `Dataset` object containing the converted samples.
    """
    sample_list = []
    for i in range(len(hf_dataset)):
        sample_list.append(
            to_plaid_sample(hf_dataset, i, flat_cst, cgns_types, enforce_shapes)
        )

    return Dataset(samples=sample_list)


def to_plaid_sample(
    ds: datasets.Dataset,
    i: int,
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
    enforce_shapes: bool = True,
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

    Notes:
        - Uses the dataset's pyarrow table data for efficient access.
        - Handles array shapes and types according to enforce_shapes.
        - Constant features from flat_cst are merged with the variable features from the row.
    """
    assert not isinstance(flat_cst[next(iter(flat_cst))], dict), (
        "did you provide the complete `flat_cst` instead of the one for the considered split?"
    )

    table = ds.data
    row = {}
    if not enforce_shapes:
        for name in table.column_names:
            value = table[name][i].values
            if value is None:
                row[name] = None  # pragma: no cover
            else:
                row[name] = value.to_numpy(zero_copy_only=False)
    else:
        for name in table.column_names:
            if isinstance(table[name][i], pa.NullScalar):
                row[name] = None  # pragma: no cover
            else:
                value = table[name][i].values
                if value is None:
                    row[name] = None  # pragma: no cover
                else:
                    if isinstance(value, pa.ListArray):
                        row[name] = np.stack(value.to_numpy(zero_copy_only=False))
                    else:
                        row[name] = value.to_numpy(zero_copy_only=True)

    flat_cst_val = {k: v for k, v in flat_cst.items() if not k.endswith("_times")}
    flat_cst_times = {k[:-6]: v for k, v in flat_cst.items() if k.endswith("_times")}

    row_val = {k: v for k, v in row.items() if not k.endswith("_times")}
    row_tim = {k[:-6]: v for k, v in row.items() if k.endswith("_times")}

    row_val.update(flat_cst_val)
    row_tim.update(flat_cst_times)

    row_val = {p: row_val[p] for p in sorted(row_val)}
    row_tim = {p: row_tim[p] for p in sorted(row_tim)}

    sample_flat_trees = {}
    paths_none = {}
    for (path_t, times_struc), (path_v, val) in zip(row_tim.items(), row_val.items()):
        assert path_t == path_v
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


def plaid_dataset_to_huggingface_datasetdict(
    dataset: Dataset,
    main_splits: dict[str, IndexType],
    processes_number: int = 1,
    writer_batch_size: int = 1,
    verbose: bool = False,
) -> tuple[datasets.DatasetDict, dict[str, Any], dict[str, Any]]:
    """Convert a PLAID dataset into a Hugging Face `datasets.DatasetDict`.

    This is a thin wrapper that creates per-split generators from a PLAID dataset
    and delegates the actual dataset construction to
    `plaid_generator_to_huggingface_datasetdict`.

    Args:
        dataset (plaid.Dataset):
            The PLAID dataset to be converted. Must support indexing with
            a list of IDs (from `main_splits`).
        main_splits (dict[str, IndexType]):
            Mapping from split names (e.g. "train", "test") to the subset of
            sample indices belonging to that split.
        processes_number (int, optional, default=1):
            Number of parallel processes to use when writing the Hugging Face dataset.
        writer_batch_size (int, optional, default=1):
            Batch size used when writing samples to disk in Hugging Face format.
        verbose (bool, optional, default=False):
            If True, print progress and debug information.

    Returns:
        datasets.DatasetDict:
            A Hugging Face `DatasetDict` containing one dataset per split.

    Example:
        >>> ds_dict = plaid_dataset_to_huggingface_datasetdict(
        ...     dataset=my_plaid_dataset,
        ...     main_splits={"train": [0, 1, 2], "test": [3]},
        ...     processes_number=4,
        ...     writer_batch_size=3
        ... )
        >>> print(ds_dict)
        DatasetDict({
            train: Dataset({
                features: ...
            }),
            test: Dataset({
                features: ...
            })
        })
    """

    def generator(dataset):
        for sample in dataset:
            yield sample

    generators = {
        split_name: partial(generator, dataset[ids])
        for split_name, ids in main_splits.items()
    }

    return plaid_generator_to_huggingface_datasetdict(
        generators, processes_number, writer_batch_size, verbose
    )


def plaid_generator_to_huggingface_datasetdict(
    generators: dict[str, Callable],
    processes_number: int = 1,
    writer_batch_size: int = 1,
    verbose: bool = False,
) -> tuple[datasets.DatasetDict, dict[str, Any], dict[str, Any]]:
    """Convert PLAID dataset generators into a Hugging Face `datasets.DatasetDict`.

    This function inspects samples produced by the given generators, flattens their
    CGNS tree structure, infers Hugging Face feature types, and builds one
    `datasets.Dataset` per split. Constant features (identical across all samples)
    are separated out from variable features.

    Args:
        generators (dict[str, Callable]):
            Mapping from split names (e.g., "train", "test") to generator functions.
            Each generator function must return an iterable of PLAID samples, where
            each sample provides `sample.features.data[0.0]` for flattening.
        processes_number (int, optional, default=1):
            Number of processes used internally by Hugging Face when materializing
            the dataset from the generators.
        writer_batch_size (int, optional, default=1):
            Batch size used when writing samples to disk in Hugging Face format.
        verbose (bool, optional, default=False):
            If True, displays progress bars and diagnostic messages.

    Returns:
        tuple:
            - **DatasetDict** (`datasets.DatasetDict`):
              A Hugging Face dataset dictionary with one dataset per split.
            - **flat_cst** (`dict[str, Any]`):
              Dictionary of constant features detected across all splits.
            - **key_mappings** (`dict[str, Any]`):
              Metadata dictionary containing:
                - `"variable_features"`: list of paths for non-constant features.
                - `"constant_features"`: list of paths for constant features.
                - `"cgns_types"`: inferred CGNS types for all features.

    Example:
        >>> ds_dict, flat_cst, key_mappings = plaid_generator_to_huggingface_datasetdict(
        ...     {"train": lambda: iter(train_samples),
        ...      "test": lambda: iter(test_samples)},
        ...     processes_number=4,
        ...     writer_batch_size=2,
        ...     verbose=True
        ... )
        >>> print(ds_dict)
        DatasetDict({
            train: Dataset({
                features: ...
            }),
            test: Dataset({
                features: ...
            })
        })
        >>> print(flat_cst)
        {'Zone1/GridCoordinates': array([0., 0.1, 0.2])}
        >>> print(key_mappings["variable_features"][:3])
        ['Zone1/FlowSolution/VelocityX', 'Zone1/FlowSolution/VelocityY', ...]
    """
    flat_cst, key_mappings, hf_features = _generator_prepare_for_huggingface(
        generators, verbose
    )

    all_features_keys = list(hf_features.keys())

    def generator_fn(gen_func, all_features_keys):
        for sample in gen_func():
            hf_sample, _, _ = build_hf_sample(sample)
            yield {path: hf_sample.get(path, None) for path in all_features_keys}

    _dict = {}
    for split_name, gen_func in generators.items():
        gen = partial(generator_fn, gen_func, all_features_keys)
        _dict[split_name] = datasets.Dataset.from_generator(
            generator=gen,
            features=hf_features,
            num_proc=processes_number,
            writer_batch_size=writer_batch_size,
            split=datasets.splits.NamedSplit(split_name),
        )

    return datasets.DatasetDict(_dict), flat_cst, key_mappings


# ------------------------------------------------------------------------------
#     HUGGING FACE HUB INTERACTIONS
# ------------------------------------------------------------------------------


def instantiate_plaid_datasetdict_from_hub(
    repo_id: str,
    enforce_shapes: bool = True,
) -> dict[str, Dataset]:  # pragma: no cover (not tested in unit tests)
    """Load a Hugging Face dataset from the Hub and instantiate it as a dictionary of PLAID datasets.

    This function retrieves a dataset dictionary from the Hugging Face Hub,
    along with its associated CGNS tree structure and type information. Each
    split of the Hugging Face dataset is then converted into a PLAID dataset.

    Args:
        repo_id (str):
            The Hugging Face repository identifier (e.g. `"user/dataset"`).
        enforce_shapes (bool, optional):
            If True, enforce strict array shapes when converting to PLAID
            datasets. Defaults to True.

    Returns:
        dict[str, Dataset]:
            A dictionary mapping split names (e.g. `"train"`, `"test"`) to
            PLAID `Dataset` objects.

    """
    hf_dataset_dict = load_dataset_from_hub(repo_id)

    flat_cst, key_mappings = load_tree_struct_from_hub(repo_id)
    cgns_types = key_mappings["cgns_types"]

    datasetdict = {}
    for split_name, hf_dataset in hf_dataset_dict.items():
        datasetdict[split_name] = to_plaid_dataset(
            hf_dataset, flat_cst, cgns_types, enforce_shapes
        )

    return datasetdict


def load_dataset_from_hub(
    repo_id: str, streaming: bool = False, *args, **kwargs
) -> Union[
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict,
]:  # pragma: no cover (not tested in unit tests)
    """Loads a Hugging Face dataset from the public hub, a private mirror, or local cache, with automatic handling of streaming and download modes.

    Behavior:

    - If the environment variable `HF_ENDPOINT` is set, uses a private Hugging Face mirror.

      - Streaming is disabled.
      - The dataset is downloaded locally via `snapshot_download` and loaded from disk.

    - If `HF_ENDPOINT` is not set, attempts to load from the public Hugging Face hub.

      - If the dataset is already cached locally, loads from disk.
      - Otherwise, loads from the hub, optionally using streaming mode.

    Args:
        repo_id (str): The Hugging Face dataset repository ID (e.g., 'username/dataset').
        streaming (bool, optional): If True, attempts to stream the dataset (only supported on the public hub).
        *args:
            Positional arguments forwarded to
            [`datasets.load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset).
        **kwargs:
            Keyword arguments forwarded to
            [`datasets.load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset).

    Returns:
        Union[datasets.Dataset, datasets.DatasetDict]: The loaded Hugging Face dataset object.

    Raises:
        Exception: Propagates any exceptions raised by `datasets.load_dataset`, `datasets.load_from_disk`, or `huggingface_hub.snapshot_download` if loading fails.

    Notes:
        - Streaming mode is not supported when using a private mirror.
        - If the dataset is found in the local cache, loads from disk instead of streaming.
        - To use behind a proxy or with a private mirror, you may need to set:
            - HF_ENDPOINT to your private mirror address
            - CURL_CA_BUNDLE to your trusted CA certificates
            - HF_HOME to a shared cache directory if needed
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()

    # Helper to check if dataset repo is already cached
    def _get_cached_path(repo_id_):
        try:
            return snapshot_download(
                repo_id=repo_id_, repo_type="dataset", local_files_only=True
            )
        except FileNotFoundError:
            return None

    # Private mirror case
    if hf_endpoint:
        if streaming:
            logger.warning(
                "Streaming mode not compatible with private mirror. Falling back to download mode."
            )
        local_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
        return load_dataset(local_path, *args, **kwargs)

    # Public case
    local_path = _get_cached_path(repo_id)
    if local_path is not None and streaming is True:
        # Even though streaming mode: rely on local files if already downloaded
        logger.info("Dataset found in cache. Loading from disk instead of streaming.")
        return load_dataset(local_path, *args, **kwargs)

    return load_dataset(repo_id, streaming=streaming, *args, **kwargs)


def load_infos_from_hub(
    repo_id: str,
) -> dict[str, dict[str, str]]:  # pragma: no cover (not tested in unit tests)
    """Load dataset infos from the Hugging Face Hub.

    Downloads the infos.yaml file from the specified repository and parses it as a dictionary.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing dataset infos.
    """
    # Download infos.yaml
    yaml_path = hf_hub_download(
        repo_id=repo_id, filename="infos.yaml", repo_type="dataset"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        infos = yaml.safe_load(f)

    return infos


def load_problem_definition_from_hub(
    repo_id: str, name: str
) -> ProblemDefinition:  # pragma: no cover (not tested in unit tests)
    """Load a ProblemDefinition from the Hugging Face Hub.

    Downloads the problem infos YAML and split JSON files from the specified repository and location,
    then initializes a ProblemDefinition object with this information.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        name (str): The name of the problem_definition stored in the repo.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    # Download splits
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"problem_definitions/{name}/train_split.json",
        repo_type="dataset",
    )
    with open(json_path, "r", encoding="utf-8") as f:
        json_data_train = json.load(f)

    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"problem_definitions/{name}/test_split.json",
        repo_type="dataset",
    )
    with open(json_path, "r", encoding="utf-8") as f:
        json_data_test = json.load(f)

    # Download problem_infos.yaml
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"problem_definitions/{name}/problem_infos.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    prob_def = ProblemDefinition()
    prob_def._initialize_from_problem_infos_dict(yaml_data)
    prob_def.set_train_split(json_data_train)
    prob_def.set_test_split(json_data_test)

    return prob_def


def load_tree_struct_from_hub(
    repo_id: str,
) -> tuple[dict, dict]:  # pragma: no cover (not tested in unit tests)
    """Load the tree structure metadata of a PLAID dataset from the Hugging Face Hub.

    This function retrieves two artifacts previously uploaded alongside a dataset:
      - **tree_constant_part.pkl**: a pickled dictionary of constant feature values
        (features that are identical across all samples).
      - **key_mappings.yaml**: a YAML file containing metadata about the dataset
        feature structure, including variable features, constant features, and CGNS types.

    Args:
        repo_id (str):
            The repository ID on the Hugging Face Hub
            (e.g., `"username/dataset_name"`).

    Returns:
        tuple[dict, dict]:
            - **flat_cst (dict)**: constant features dictionary (path → value).
            - **key_mappings (dict)**: metadata dictionary containing keys such as:
                - `"variable_features"`: list of paths for non-constant features.
                - `"constant_features"`: list of paths for constant features.
                - `"cgns_types"`: mapping from paths to CGNS types.
    """
    # constant part of the tree
    flat_cst_path = hf_hub_download(
        repo_id=repo_id,
        filename="tree_constant_part.pkl",
        repo_type="dataset",
    )

    with open(flat_cst_path, "rb") as f:
        flat_cst = pickle.load(f)

    # key mappings
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename="key_mappings.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        key_mappings = yaml.safe_load(f)

    return flat_cst, key_mappings


def push_dataset_dict_to_hub(
    repo_id: str, hf_dataset_dict: datasets.DatasetDict, *args, **kwargs
) -> None:  # pragma: no cover (not tested in unit tests)
    """Push a Hugging Face `DatasetDict` to the Hugging Face Hub.

    This is a thin wrapper around `datasets.DatasetDict.push_to_hub`, allowing
    you to upload a dataset dictionary (with one or more splits such as
    `"train"`, `"validation"`, `"test"`) to the Hugging Face Hub.

    Args:
        repo_id (str):
            The repository ID on the Hugging Face Hub
            (e.g. `"username/dataset_name"`).
        hf_dataset_dict (datasets.DatasetDict):
            The Hugging Face dataset dictionary to push.
        *args:
            Positional arguments forwarded to
            [`DatasetDict.push_to_hub`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub).
        **kwargs:
            Keyword arguments forwarded to
            [`DatasetDict.push_to_hub`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub).

    Returns:
        None
    """
    hf_dataset_dict.push_to_hub(repo_id, *args, **kwargs)


def push_infos_to_hub(
    repo_id: str, infos: dict[str, dict[str, str]]
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload dataset infos to the Hugging Face Hub.

    Serializes the infos dictionary to YAML and uploads it to the specified repository as infos.yaml.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset infos to upload.

    Raises:
        ValueError: If the infos dictionary is empty.
    """
    if len(infos) > 0:
        api = HfApi()
        yaml_str = yaml.dump(infos)
        yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))
        api.upload_file(
            path_or_fileobj=yaml_buffer,
            path_in_repo="infos.yaml",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload infos.yaml",
        )
    else:
        raise ValueError("'infos' must not be empty")


def push_problem_definition_to_hub(
    repo_id: str, name: str, pb_def: ProblemDefinition
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload a ProblemDefinition and its split information to the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        name (str): The name of the problem_definition to store in the repo.
        pb_def (ProblemDefinition): The problem definition to upload.
    """
    api = HfApi()
    data = pb_def._generate_problem_infos_dict()
    for k, v in list(data.items()):
        if not v:
            data.pop(k)
    if data is not None:
        yaml_str = yaml.dump(data)
        yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo=f"problem_definitions/{name}/problem_infos.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload problem_definitions/{name}/problem_infos.yaml",
    )

    # data = pb_def.get_split()
    # json_str = json.dumps(data)
    # json_buffer = io.BytesIO(json_str.encode("utf-8"))

    # api.upload_file(
    #     path_or_fileobj=json_buffer,
    #     path_in_repo=f"problem_definitions/{name}/split.json",
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     commit_message=f"Upload problem_definitions/{name}/split.json",
    # )

    # data = pb_def.get_train_split()
    # json_str = json.dumps(data)
    # json_buffer = io.BytesIO(json_str.encode("utf-8"))

    # api.upload_file(
    #     path_or_fileobj=json_buffer,
    #     path_in_repo=f"problem_definitions/{name}/train_split.json",
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     commit_message=f"Upload problem_definitions/{name}/train_split.json",
    # )

    # data = pb_def.get_test_split()
    # json_str = json.dumps(data)
    # json_buffer = io.BytesIO(json_str.encode("utf-8"))

    # api.upload_file(
    #     path_or_fileobj=json_buffer,
    #     path_in_repo=f"problem_definitions/{name}/test_split.json",
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     commit_message=f"Upload problem_definitions/{name}/test_split.json",
    # )


def push_tree_struct_to_hub(
    repo_id: str,
    flat_cst: dict[str, Any],
    key_mappings: dict[str, Any],
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload a dataset's tree structure to a Hugging Face dataset repository.

    This function pushes two components of a dataset tree structure to the specified
    Hugging Face Hub repository:

    1. `flat_cst`: the constant parts of the dataset tree, serialized as a pickle file
       (`tree_constant_part.pkl`).
    2. `key_mappings`: the dictionary of key mappings and metadata for the dataset tree,
       serialized as a YAML file (`key_mappings.yaml`).

    Both files are uploaded using the Hugging Face `HfApi().upload_file` method.

    Args:
        repo_id (str): The Hugging Face dataset repository ID where files will be uploaded.
        flat_cst (dict[str, Any]): Dictionary containing constant values in the dataset tree.
        key_mappings (dict[str, Any]): Dictionary containing key mappings and additional metadata.

    Returns:
        None

    Notes:
        - Each upload includes a commit message indicating the filename.
        - This function is not covered by unit tests (`pragma: no cover`).
    """
    api = HfApi()

    # constant part of the tree
    api.upload_file(
        path_or_fileobj=io.BytesIO(pickle.dumps(flat_cst)),
        path_in_repo="tree_constant_part.pkl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload tree_constant_part.pkl",
    )

    # key mappings
    yaml_str = yaml.dump(key_mappings, sort_keys=False)
    yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo="key_mappings.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload key_mappings.yaml",
    )


# ------------------------------------------------------------------------------
#     HUGGING FACE INTERACTIONS ON DISK
# ------------------------------------------------------------------------------


def load_dataset_from_disk(
    path: Union[str, Path], *args, **kwargs
) -> Union[datasets.Dataset, datasets.DatasetDict]:
    """Load a Hugging Face dataset or dataset dictionary from disk.

    This function wraps `datasets.load_from_disk` to accept either a string path or a
    `Path` object and returns the loaded dataset object.

    Args:
        path (Union[str, Path]): Path to the directory containing the saved dataset.
        *args:
            Positional arguments forwarded to
            [`datasets.load_from_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_from_disk).
        **kwargs:
            Keyword arguments forwarded to
            [`datasets.load_from_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_from_disk).

    Returns:
        Union[datasets.Dataset, datasets.DatasetDict]: The loaded Hugging Face dataset
        object, which may be a single `Dataset` or a `DatasetDict` depending on what
        was saved on disk.
    """
    return load_from_disk(str(path), *args, **kwargs)


def load_infos_from_disk(path: Union[str, Path]) -> dict[str, dict[str, str]]:
    """Load dataset information from a YAML file stored on disk.

    Args:
        path (Union[str, Path]): Directory path containing the `infos.yaml` file.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing dataset infos.
    """
    infos_fname = Path(path) / "infos.yaml"
    with infos_fname.open("r") as file:
        infos = yaml.safe_load(file)
    return infos


def load_problem_definition_from_disk(
    path: Union[str, Path], name: Union[str, Path]
) -> ProblemDefinition:
    """Load a ProblemDefinition and its split information from disk.

    Args:
        path (Union[str, Path]): The root directory path for loading.
        name (str): The name of the problem_definition stored in the disk directory.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    pb_def = ProblemDefinition()
    pb_def._load_from_file_(Path(path) / Path("problem_definitions") / Path(name))
    return pb_def


def load_tree_struct_from_disk(
    path: Union[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a tree structure for a dataset from disk.

    This function loads two components from the specified directory:
    1. `tree_constant_part.pkl`: a pickled dictionary containing the constant parts of the tree.
    2. `key_mappings.yaml`: a YAML file containing key mappings and metadata.

    Args:
        path (Union[str, Path]): Directory path containing the `tree_constant_part.pkl`
            and `key_mappings.yaml` files.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - `flat_cst` (dict): Dictionary of constant tree values.
            - `key_mappings` (dict): Dictionary of key mappings and metadata.
    """
    with open(Path(path) / Path("key_mappings.yaml"), "r", encoding="utf-8") as f:
        key_mappings = yaml.safe_load(f)

    with open(Path(path) / "tree_constant_part.pkl", "rb") as f:
        flat_cst = pickle.load(f)

    return flat_cst, key_mappings


def save_dataset_dict_to_disk(
    path: Union[str, Path], hf_dataset_dict: datasets.DatasetDict, *args, **kwargs
) -> None:
    """Save a Hugging Face DatasetDict to disk.

    This function serializes the provided DatasetDict and writes it to the specified
    directory, preserving its features, splits, and data for later loading.

    Args:
        path (Union[str, Path]): Directory path where the DatasetDict will be saved.
        hf_dataset_dict (datasets.DatasetDict): The Hugging Face DatasetDict to save.
        *args:
            Positional arguments forwarded to
            [`DatasetDict.save_to_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.save_to_disk).
        **kwargs:
            Keyword arguments forwarded to
            [`DatasetDict.save_to_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.save_to_disk).

    Returns:
        None
    """
    hf_dataset_dict.save_to_disk(str(path), *args, **kwargs)


def save_infos_to_disk(
    path: Union[str, Path], infos: dict[str, dict[str, str]]
) -> None:
    """Save dataset infos as a YAML file to disk.

    Args:
        path (Union[str, Path]): The directory path where the infos file will be saved.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset infos.
    """
    infos_fname = Path(path) / "infos.yaml"
    infos_fname.parent.mkdir(parents=True, exist_ok=True)
    with open(infos_fname, "w") as file:
        yaml.dump(infos, file, default_flow_style=False, sort_keys=False)


def save_problem_definition_to_disk(
    path: Union[str, Path], name: Union[str, Path], pb_def: ProblemDefinition
) -> None:
    """Save a ProblemDefinition and its split information to disk.

    Args:
        path (Union[str, Path]): The root directory path for saving.
        name (str): The name of the problem_definition to store in the disk directory.
        pb_def (ProblemDefinition): The problem definition to save.
    """
    pb_def._save_to_file_(Path(path) / Path("problem_definitions") / Path(name))


#         pbdef_fname = path / "problem_infos.yaml"
#         with pbdef_fname.open("w") as file:
#             yaml.dump(
#                 problem_infos_dict, file, default_flow_style=False, sort_keys=False
#             )


# def push_problem_definition_to_hub(
#     repo_id: str, name: str, pb_def: ProblemDefinition
# ) -> None:  # pragma: no cover (not tested in unit tests)
#     """Upload a ProblemDefinition and its split information to the Hugging Face Hub.

#     Args:
#         repo_id (str): The repository ID on the Hugging Face Hub.
#         name (str): The name of the problem_definition to store in the repo.
#         pb_def (ProblemDefinition): The problem definition to upload.
#     """
#     api = HfApi()
#     data = pb_def._generate_problem_infos_dict()
#     for k, v in list(data.items()):
#         if not v:
#             data.pop(k)
#     if data is not None:
#         yaml_str = yaml.dump(data)
#         yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

#     api.upload_file(
#         path_or_fileobj=yaml_buffer,
#         path_in_repo=f"problem_definitions/{name}/problem_infos.yaml",
#         repo_id=repo_id,
#         repo_type="dataset",
#         commit_message=f"Upload problem_definitions/{name}/problem_infos.yaml",
#     )


def save_tree_struct_to_disk(
    path: Union[str, Path],
    flat_cst: dict[str, Any],
    key_mappings: dict[str, Any],
) -> None:
    """Save the structure of a dataset tree to disk.

    This function writes the constant part of the tree and its key mappings to files
    in the specified directory. The constant part is serialized as a pickle file,
    while the key mappings are saved in YAML format.

    Args:
        path (Union[str, Path]): Directory path where the tree structure files will be saved.
        flat_cst (dict): Dictionary containing the constant part of the tree.
        key_mappings (dict): Dictionary containing key mappings for the tree structure.

    Returns:
        None
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(Path(path) / "tree_constant_part.pkl", "wb") as f:  # wb = write binary
        pickle.dump(flat_cst, f)

    with open(Path(path) / "key_mappings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(key_mappings, f, sort_keys=False)


# ------------------------------------------------------------------------------
#     HUGGING FACE BINARY BRIDGE
# ------------------------------------------------------------------------------


def binary_to_plaid_sample(hf_sample: dict[str, bytes]) -> Sample:
    """Convert a Hugging Face dataset sample in binary format to a Plaid `Sample`.

    The input `hf_sample` is expected to contain a pickled representation of a sample
    under the key `"sample"`. This function attempts to validate the unpickled sample
    as a Plaid `Sample`. If validation fails, it reconstructs the sample from its
    components (`meshes`, `path`, and optional `scalars`) before validating it.

    Args:
        hf_sample (dict[str, bytes]): A dictionary representing a Hugging Face sample,
            with the pickled sample stored under the key `"sample"`.

    Returns:
        Sample: A validated Plaid `Sample` object.

    Raises:
        KeyError: If required keys (`"sample"`, `"meshes"`, `"path"`) are missing
            and the sample cannot be reconstructed.
        ValidationError: If the reconstructed sample still fails Plaid validation.
    """
    pickled_hf_sample = pickle.loads(hf_sample["sample"])

    try:
        # Try to validate the sample
        return Sample.model_validate(pickled_hf_sample)

    except ValidationError:
        features = SampleFeatures(
            data=pickled_hf_sample.get("meshes"),
        )

        sample = Sample(
            path=pickled_hf_sample.get("path"),
            features=features,
        )

        scalars = pickled_hf_sample.get("scalars")
        if scalars:
            for sn, val in scalars.items():
                sample.add_scalar(sn, val)

        return Sample.model_validate(sample)


def plaid_dataset_to_huggingface_binary(
    dataset: Dataset,
    ids: Optional[list[IndexType]] = None,
    split_name: str = "all_samples",
    processes_number: int = 1,
) -> datasets.Dataset:
    """Use this function for converting a Hugging Face dataset from a plaid dataset.

    The dataset can then be saved to disk, or pushed to the Hugging Face hub.

    Args:
        dataset (Dataset): the plaid dataset to be converted in Hugging Face format
        ids (list, optional): The specific sample IDs to convert the dataset. Defaults to None.
        split_name (str): The name of the split. Default: "all_samples".
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.Dataset: dataset in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_dataset_to_huggingface_binary(dataset, problem_definition, split)
            dataset.save_to_disk("path/to/dir)
            dataset.push_to_hub("chanel/dataset")
    """
    if ids is None:
        ids = dataset.get_sample_ids()

    def generator():
        for sample in dataset[ids]:
            yield {
                "sample": pickle.dumps(sample.model_dump()),
            }

    return plaid_generator_to_huggingface_binary(
        generator=generator,
        split_name=split_name,
        processes_number=processes_number,
    )


def plaid_generator_to_huggingface_binary(
    generator: Callable,
    split_name: str = "all_samples",
    processes_number: int = 1,
) -> datasets.Dataset:
    """Use this function for creating a Hugging Face dataset from a sample generator function.

    This function can be used when the plaid dataset cannot be loaded in RAM all at once due to its size.
    The generator enables loading samples one by one.

    Args:
        generator (Callable): a function yielding a dict {"sample" : sample}, where sample is of type 'bytes'
        split_name (str): The name of the split. Default: "all_samples".
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.Dataset: dataset in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_generator_to_huggingface_binary(generator, infos, split)
    """
    ds: datasets.Dataset = datasets.Dataset.from_generator(  # pyright: ignore[reportAssignmentType]
        generator=generator,
        features=datasets.Features({"sample": datasets.Value("binary")}),
        num_proc=processes_number,
        writer_batch_size=1,
        split=datasets.splits.NamedSplit(split_name),
    )

    return ds


def plaid_dataset_to_huggingface_datasetdict_binary(
    dataset: Dataset,
    main_splits: dict[str, IndexType],
    processes_number: int = 1,
) -> datasets.DatasetDict:
    """Use this function for converting a Hugging Face dataset dict from a plaid dataset.

    The dataset can then be saved to disk, or pushed to the Hugging Face hub.

    Args:
        dataset (Dataset): the plaid dataset to be converted in Hugging Face format.
        main_splits (list[str]): The name of the main splits: defining a partitioning of the sample ids.
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.Dataset: dataset in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_dataset_to_huggingface_datasetdict_binary(dataset, problem_definition, split)
            dataset.save_to_disk("path/to/dir)
            dataset.push_to_hub("chanel/dataset")
    """
    _dict = {}
    for split_name, ids in main_splits.items():
        ds = plaid_dataset_to_huggingface_binary(
            dataset=dataset,
            ids=ids,
            processes_number=processes_number,
        )
        _dict[split_name] = ds

    return datasets.DatasetDict(_dict)


def plaid_generator_to_huggingface_datasetdict_binary(
    generators: dict[str, Callable],
    processes_number: int = 1,
) -> datasets.DatasetDict:
    """Use this function for creating a Hugging Face dataset dict (containing multiple splits) from a sample generator function.

    This function can be used when the plaid dataset cannot be loaded in RAM all at once due to its size.
    The generator enables loading samples one by one.
    The dataset dict can then be saved to disk, or pushed to the Hugging Face hub.

    Notes:
        Only the first split will contain the decription.

    Args:
        generators (dict[str, Callable]): a dict of functions yielding a dict {"sample" : sample}, where sample is of type 'bytes'
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.DatasetDict: dataset dict in Hugging Face format

    Example:
        .. code-block:: python

            hf_dataset_dict = plaid_generator_to_huggingface_datasetdict(generator, infos, problem_definition, main_splits)
            push_dataset_dict_to_hub("chanel/dataset", hf_dataset_dict)
            hf_dataset_dict.save_to_disk("path/to/dir")
    """
    _dict = {}
    for split_name, generator in generators.items():
        ds = plaid_generator_to_huggingface_binary(
            generator=generator,
            processes_number=processes_number,
            split_name=split_name,
        )
        _dict[split_name] = ds

    return datasets.DatasetDict(_dict)


def huggingface_dataset_to_plaid(
    ds: datasets.Dataset,
    ids: Optional[list[int]] = None,
    processes_number: int = 1,
    large_dataset: bool = False,
    verbose: bool = True,
) -> Union[Dataset, ProblemDefinition]:
    """Use this function for converting a plaid dataset from a Hugging Face dataset.

    A Hugging Face dataset can be read from disk or the hub. From the hub, the
    split = "all_samples" options is important to get a dataset and not a datasetdict.
    Many options from loading are available (caching, streaming, etc...)

    Args:
        ds (datasets.Dataset): the dataset in Hugging Face format to be converted
        ids (list, optional): The specific sample IDs to load from the dataset. Defaults to None.
        processes_number (int, optional): The number of processes used to generate the plaid dataset
        large_dataset (bool): if True, uses a variant where parallel worker do not each load the complete dataset. Default: False.
        verbose (bool, optional): if True, prints progress using tdqm

    Returns:
        dataset (Dataset): the converted dataset.
        problem_definition (ProblemDefinition): the problem definition generated from the Hugging Face dataset

    Example:
        .. code-block:: python

            from datasets import load_dataset, load_from_disk

            dataset = load_dataset("path/to/dir", split = "all_samples")
            dataset = load_from_disk("chanel/dataset")
            plaid_dataset, plaid_problem = huggingface_dataset_to_plaid(dataset)
    """
    from plaid.bridges.huggingface_helpers import (
        _HFShardToPlaidSampleConverter,
        _HFToPlaidSampleConverter,
    )

    assert processes_number <= len(ds), (
        "Trying to parallelize with more processes than samples in dataset"
    )
    if ids:
        assert processes_number <= len(ids), (
            "Trying to parallelize with more processes than selected samples in dataset"
        )

    description = "Converting Hugging Face binary dataset to plaid"

    dataset = Dataset()

    if large_dataset:
        if ids:
            raise NotImplementedError(
                "ids selection not implemented with large_dataset option"
            )
        for i in range(processes_number):
            shard = ds.shard(num_shards=processes_number, index=i)
            shard.save_to_disk(f"./shards/dataset_shard_{i}")

        def parallel_convert(shard_path, n_workers):
            converter = _HFShardToPlaidSampleConverter(shard_path)
            with Pool(processes=n_workers) as pool:
                return list(
                    tqdm(
                        pool.imap(converter, range(len(converter.hf_ds))),
                        total=len(converter.hf_ds),
                        disable=not verbose,
                        desc=description,
                    )
                )

        samples = []

        for i in range(processes_number):
            shard_path = Path(".") / "shards" / f"dataset_shard_{i}"
            shard_samples = parallel_convert(shard_path, n_workers=processes_number)
            samples.extend(shard_samples)

        dataset.add_samples(samples, ids)

        shards_dir = Path(".") / "shards"
        if shards_dir.exists() and shards_dir.is_dir():
            shutil.rmtree(shards_dir)

    else:
        if ids:
            indices = ids
        else:
            indices = range(len(ds))

        if processes_number == 1:
            for idx in tqdm(
                indices, total=len(indices), disable=not verbose, desc=description
            ):
                sample = _HFToPlaidSampleConverter(ds)(idx)
                dataset.add_sample(sample, id=idx)

        else:
            with Pool(processes=processes_number) as pool:
                for idx, sample in enumerate(
                    tqdm(
                        pool.imap(_HFToPlaidSampleConverter(ds), indices),
                        total=len(indices),
                        disable=not verbose,
                        desc=description,
                    )
                ):
                    dataset.add_sample(sample, id=indices[idx])

    infos = huggingface_description_to_infos(ds.description)

    dataset.set_infos(infos)

    problem_definition = huggingface_description_to_problem_definition(ds.description)

    return dataset, problem_definition


def huggingface_description_to_problem_definition(
    description: dict,
) -> ProblemDefinition:
    """Converts a Hugging Face dataset description to a plaid problem definition.

    Args:
        description (dict): the description field of a Hugging Face dataset, containing the problem definition

    Returns:
        problem_definition (ProblemDefinition): the plaid problem definition initialized from the Hugging Face dataset description
    """
    description = {} if description == "" else description
    problem_definition = ProblemDefinition()
    for func, key in [
        (problem_definition.set_task, "task"),
        (problem_definition.set_split, "split"),
        (problem_definition.add_input_scalars_names, "in_scalars_names"),
        (problem_definition.add_output_scalars_names, "out_scalars_names"),
        (problem_definition.add_input_fields_names, "in_fields_names"),
        (problem_definition.add_output_fields_names, "out_fields_names"),
        (problem_definition.add_input_meshes_names, "in_meshes_names"),
        (problem_definition.add_output_meshes_names, "out_meshes_names"),
    ]:
        try:
            func(description[key])
        except KeyError:
            logger.info(f"Could not retrieve key:'{key}' from description")
            pass

    return problem_definition


def huggingface_description_to_infos(
    description: dict,
) -> dict[str, dict[str, str]]:
    """Convert a Hugging Face dataset description dictionary to a PLAID infos dictionary.

    Extracts the "legal" and "data_production" sections from the Hugging Face description
    and returns them in a format compatible with PLAID dataset infos.

    Args:
        description (dict): The Hugging Face dataset description dictionary.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing "legal" and "data_production" infos if present.
    """
    infos = {}
    if "legal" in description:
        infos["legal"] = description["legal"]
    if "data_production" in description:
        infos["data_production"] = description["data_production"]
    return infos


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.9",
    removal="0.2.0",
)
def create_string_for_huggingface_dataset_card(
    description: dict,
    download_size_bytes: int,
    dataset_size_bytes: int,
    nb_samples: int,
    owner: str,
    license: str,
    zenodo_url: Optional[str] = None,
    arxiv_paper_url: Optional[str] = None,
    pretty_name: Optional[str] = None,
    size_categories: Optional[list[str]] = None,
    task_categories: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    dataset_long_description: Optional[str] = None,
    url_illustration: Optional[str] = None,
) -> str:
    """Use this function for creating a dataset card, to upload together with the datase on the Hugging Face hub.

    Doing so ensure that load_dataset from the hub will populate the hf-dataset.description field, and be compatible for conversion to plaid.

    Without a dataset_card, the description field is lost.

    The parameters download_size_bytes and dataset_size_bytes can be determined after a
    dataset has been uploaded on Hugging Face:
    - manually by reading their values on the dataset page README.md,
    - automatically as shown in the example below

    See `the hugginface examples <https://github.com/PLAID-lib/plaid/blob/main/examples/bridges/huggingface_bridge_example.py>`__ for a concrete use.

    Args:
        description (dict): Hugging Face dataset description. Obtained from
        - description = hf_dataset.description
        - description = generate_huggingface_description(infos, problem_definition)
        download_size_bytes (int): the size of the dataset when downloaded from the hub
        dataset_size_bytes (int): the size of the dataset when loaded in RAM
        nb_samples (int): the number of samples in the dataset
        owner (str): the owner of the dataset, usually a username or organization name on Hugging Face
        license (str): the license of the dataset, e.g. "CC-BY-4.0", "CC0-1.0", etc.
        zenodo_url (str, optional): the Zenodo URL of the dataset, if available
        arxiv_paper_url (str, optional): the arxiv paper URL of the dataset, if available
        pretty_name (str, optional): a human-readable name for the dataset, e.g. "PLAID Dataset"
        size_categories (list[str], optional): size categories of the dataset, e.g. ["small", "medium", "large"]
        task_categories (list[str], optional): task categories of the dataset, e.g. ["image-classification", "text-generation"]
        tags (list[str], optional): tags for the dataset, e.g. ["3D", "simulation", "mesh"]
        dataset_long_description (str, optional): a long description of the dataset, providing more details about its content and purpose
        url_illustration (str, optional): a URL to an illustration image for the dataset, e.g. a screenshot or a sample mesh

    Returns:
        dataset (Dataset): the converted dataset
        problem_definition (ProblemDefinition): the problem definition generated from the Hugging Face dataset

    Example:
        .. code-block:: python

            hf_dataset.push_to_hub("chanel/dataset")

            from datasets import load_dataset_builder

            datasetInfo = load_dataset_builder("chanel/dataset").__getstate__()['info']

            from huggingface_hub import DatasetCard

            card_text = create_string_for_huggingface_dataset_card(
                description = description,
                download_size_bytes = datasetInfo.download_size,
                dataset_size_bytes = datasetInfo.dataset_size,
                ...)
            dataset_card = DatasetCard(card_text)
            dataset_card.push_to_hub("chanel/dataset")
    """
    str__ = f"""---
license: {license}
"""

    if size_categories:
        str__ += f"""size_categories:
  {size_categories}
"""

    if task_categories:
        str__ += f"""task_categories:
  {task_categories}
"""

    if pretty_name:
        str__ += f"""pretty_name: {pretty_name}
"""

    if tags:
        str__ += f"""tags:
  {tags}
"""

    str__ += f"""configs:
  - config_name: default
    data_files:
      - split: all_samples
        path: data/all_samples-*
dataset_info:
  description: {description}
  features:
  - name: sample
    dtype: binary
  splits:
  - name: all_samples
    num_bytes: {dataset_size_bytes}
    num_examples: {nb_samples}
  download_size: {download_size_bytes}
  dataset_size: {dataset_size_bytes}
---

# Dataset Card
"""
    if url_illustration:
        str__ += f"""![image/png]({url_illustration})

This dataset contains a single Hugging Face split, named 'all_samples'.

The samples contains a single Hugging Face feature, named called "sample".

Samples are instances of [plaid.containers.sample.Sample](https://plaid-lib.readthedocs.io/en/latest/autoapi/plaid/containers/sample/index.html#plaid.containers.sample.Sample).
Mesh objects included in samples follow the [CGNS](https://cgns.github.io/) standard, and can be converted in
[Muscat.Containers.Mesh.Mesh](https://muscat.readthedocs.io/en/latest/_source/Muscat.Containers.Mesh.html#Muscat.Containers.Mesh.Mesh).


Example of commands:
```python
import pickle
from datasets import load_dataset
from plaid import Sample

# Load the dataset
dataset = load_dataset("chanel/dataset", split="all_samples")

# Get the first sample of the first split
split_names = list(dataset.description["split"].keys())
ids_split_0 = dataset.description["split"][split_names[0]]
sample_0_split_0 = dataset[ids_split_0[0]]["sample"]
plaid_sample = Sample.model_validate(pickle.loads(sample_0_split_0))
print("type(plaid_sample) =", type(plaid_sample))

print("plaid_sample =", plaid_sample)

# Get a field from the sample
field_names = plaid_sample.get_field_names()
field = plaid_sample.get_field(field_names[0])
print("field_names[0] =", field_names[0])

print("field.shape =", field.shape)

# Get the mesh and convert it to Muscat
from Muscat.Bridges import CGNSBridge
CGNS_tree = plaid_sample.get_mesh()
mesh = CGNSBridge.CGNSToMesh(CGNS_tree)
print(mesh)
```

## Dataset Details

### Dataset Description

"""

    if dataset_long_description:
        str__ += f"""{dataset_long_description}
"""

    str__ += f"""- **Language:** [PLAID](https://plaid-lib.readthedocs.io/)
- **License:** {license}
- **Owner:** {owner}
"""

    if zenodo_url or arxiv_paper_url:
        str__ += """
### Dataset Sources

"""

    if zenodo_url:
        str__ += f"""- **Repository:** [Zenodo]({zenodo_url})
"""

    if arxiv_paper_url:
        str__ += f"""- **Paper:** [arxiv]({arxiv_paper_url})
"""

    return str__
