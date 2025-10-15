"""Hugging Face bridge for PLAID datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
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
from plaid.types.cgns_types import CGNSTree
from plaid.utils.cgns_helper import flatten_cgns_tree, unflatten_cgns_tree
from plaid.utils.deprecation import deprecated

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
#     HUGGING FACE BRIDGE (with tree flattening and pyarrow tables)
# ------------------------------------------------------------------------------


def to_cgns_tree_columnar(
    ds: datasets.Dataset,
    i: int,
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
    enforce_shapes: bool = False,
) -> CGNSTree:
    """Convert a Hugging Face dataset row to a PLAID Sample object.

    This function extracts a single row from a Hugging Face dataset and converts it
    into a PLAID Sample by unflattening the CGNS tree structure. Constant features
    are added from the flat_cst dictionary.

    Args:
        ds (datasets.Dataset): The Hugging Face dataset containing the sample data.
        i (int): The index of the row to convert.
        flat_cst (dict[str, any]): Dictionary of constant features to add to each sample.
        cgns_types (dict[str, str]): Dictionary mapping paths to CGNS types for reconstruction.
        enforce_shapes (bool, optional): If True, ensures consistent array shapes during conversion.
            Defaults to False.

    Returns:
        Sample: A validated PLAID Sample object reconstructed from the Hugging Face dataset row.

    Notes:
        - The function uses the dataset's pyarrow table data for efficient access
        - When enforce_shapes is False, it uses zero_copy_only=False for numpy conversion
        - When enforce_shapes is True, it handles ListArray types specially by stacking them
        - Constant features from flat_cst are merged with the variable features from the row
    """
    table = ds.data

    row = {}
    if not enforce_shapes:
        for name in table.column_names:
            value = table[name][i].values
            if value is None:
                row[name] = None
            else:
                row[name] = value.to_numpy(zero_copy_only=False)
    else:
        for name in table.column_names:
            value = table[name][i].values
            if value is None:
                row[name] = None
            else:
                if isinstance(value, pa.ListArray):
                    row[name] = np.stack(value.to_numpy(zero_copy_only=False))
                else:
                    row[name] = value.to_numpy(zero_copy_only=True)

    row.update(flat_cst)
    return unflatten_cgns_tree(row, cgns_types)


def to_cgns_tree(
    hf_sample: dict[str, Features], flat_cst: dict[str, Any], cgns_types: dict[str, str]
) -> CGNSTree:
    """Convert a Hugging Face dataset row to a PLAID Sample object.

    This function extracts a single row from a Hugging Face dataset and converts it
    into a PLAID Sample by unflattening the CGNS tree structure. Constant features
    are added from the flat_cst dictionary.

    Args:
        hf_sample (dict[str, Features]): row of a Hugging Face dataset
        flat_cst (dict[str, any]): Dictionary of constant features to add to each sample.
        cgns_types (dict[str, str]): Dictionary mapping paths to CGNS types for reconstruction.
        enforce_shapes (bool, optional): If True, ensures consistent array shapes during conversion.
            Defaults to False.

    Returns:
        Sample: A validated PLAID Sample object reconstructed from the Hugging Face dataset row.

    Notes:
        - The function uses the dataset's pyarrow table data for efficient access
        - When enforce_shapes is False, it uses zero_copy_only=False for numpy conversion
        - When enforce_shapes is True, it handles ListArray types specially by stacking them
        - Constant features from flat_cst are merged with the variable features from the row
    """
    row = {name: np.array(value) for name, value in hf_sample.items()}
    row.update(flat_cst)
    return unflatten_cgns_tree(row, cgns_types)


def to_plaid_sample_columnar(
    ds: datasets.Dataset,
    i: int,
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
    enforce_shapes: bool = False,
) -> Sample:
    """Convert a Hugging Face dataset row to a PLAID Sample object.

    This function extracts a single row from a Hugging Face dataset and converts it
    into a PLAID Sample by unflattening the CGNS tree structure. Constant features
    are added from the flat_cst dictionary.

    Args:
        ds (datasets.Dataset): The Hugging Face dataset containing the sample data.
        i (int): The index of the row to convert.
        flat_cst (dict[str, any]): Dictionary of constant features to add to each sample.
        cgns_types (dict[str, str]): Dictionary mapping paths to CGNS types for reconstruction.
        enforce_shapes (bool, optional): If True, ensures consistent array shapes during conversion.
            Defaults to False.

    Returns:
        Sample: A validated PLAID Sample object reconstructed from the Hugging Face dataset row.

    Notes:
        - The function uses the dataset's pyarrow table data for efficient access
        - When enforce_shapes is False, it uses zero_copy_only=False for numpy conversion
        - When enforce_shapes is True, it handles ListArray types specially by stacking them
        - Constant features from flat_cst are merged with the variable features from the row
    """
    cgns_tree = to_cgns_tree_columnar(ds, i, flat_cst, cgns_types, enforce_shapes)

    sample = Sample(path=None, features=SampleFeatures({0.0: cgns_tree}))
    return Sample.model_validate(sample)


def to_plaid_sample(
    hf_sample: dict[str, Features],
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
) -> Sample:
    """Convert a Hugging Face dataset row to a PLAID Sample object.

    This function extracts a single row from a Hugging Face dataset and converts it
    into a PLAID Sample by unflattening the CGNS tree structure. Constant features
    are added from the flat_cst dictionary.

    Args:
        hf_sample (dict[str, Features]): row of a Hugging Face dataset
        flat_cst (dict[str, any]): Dictionary of constant features to add to each sample.
        cgns_types (dict[str, str]): Dictionary mapping paths to CGNS types for reconstruction.

    Returns:
        Sample: A validated PLAID Sample object reconstructed from the Hugging Face dataset row.

    Notes:
        - The function uses the dataset's pyarrow table data for efficient access
        - When enforce_shapes is False, it uses zero_copy_only=False for numpy conversion
        - When enforce_shapes is True, it handles ListArray types specially by stacking them
        - Constant features from flat_cst are merged with the variable features from the row
    """
    cgns_tree = to_cgns_tree(hf_sample, flat_cst, cgns_types)

    sample = Sample(path=None, features=SampleFeatures({0.0: cgns_tree}))
    return Sample.model_validate(sample)


def to_plaid_dataset(
    hf_dataset: datasets.Dataset,
    flat_cst: dict[str, Any],
    cgns_types: dict[str, str],
    enforce_shapes: bool = True,
) -> Dataset:
    """Convert a Hugging Face dataset into a PLAID dataset.

    Iterates over all samples in a Hugging Face `Dataset` and converts each one
    into a PLAID-compatible sample using `to_plaid_sample_columnar`. The resulting
    samples are then collected into a single PLAID `Dataset`.

    Args:
        hf_dataset (datasets.Dataset):
            The Hugging Face dataset split to convert.
        flat_cst:
            Flattened representation of the CGNS tree structure constants,
            used to map data fields.
        cgns_types:
            Mapping of CGNS paths to their expected types.
        enforce_shapes (bool, optional):
            If True, ensures all arrays strictly follow the reference shapes.
            Defaults to False.

    Returns:
        Dataset:
            A PLAID `Dataset` object containing the converted samples.

    """
    sample_list = []
    for i in range(len(hf_dataset)):
        sample_list.append(
            to_plaid_sample_columnar(
                hf_dataset, i, flat_cst, cgns_types, enforce_shapes
            )
        )

    return Dataset(samples=sample_list)


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
        return Value("null")

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
        else:
            return Value("string")

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


def _generator_prepare_for_huggingface(
    generators: dict[str, Callable],
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], Features]:
    """Inspect PLAID dataset generators and infer Hugging Face feature schema.

    This function scans all provided split generators to:
      1. Flatten each CGNS tree into a dictionary of paths → values.
      2. Infer Hugging Face `Features` types for all variable leaves.
      3. Detect constant leaves (values that never change across all samples).
      4. Collect global CGNS type metadata.

    Args:
        generators (dict[str, Callable]):
            A dictionary mapping split names to callables returning sample generators.
            Each sample is expected to have the structure `sample.features.data[0.0]`
            compatible with `flatten_cgns_tree`.
        verbose (bool, optional, default=True):
            If True, displays progress bars while processing splits.

    Returns:
        tuple:
            - **flat_cst (dict[str, Any])**:
              Mapping from feature path to constant values detected across all splits.
            - **key_mappings (dict[str, Any])**:
              Metadata dictionary with:
                - `"variable_features"` (list[str]): paths of non-constant features.
                - `"constant_features"` (list[str]): paths of constant features.
                - `"cgns_types"` (dict[str, Any]): CGNS type information for all paths.
            - **hf_features (datasets.Features)**:
              Hugging Face feature specification for variable features.

    Raises:
        ValueError:
            If inconsistent CGNS types or feature types are found for the same path.

    Example:
        >>> flat_cst, key_mappings, hf_features = _generator_prepare_for_huggingface(
        ...     {"train": lambda: iter(train_samples),
        ...      "test": lambda: iter(test_samples)}
        ... )
        >>> print(key_mappings["variable_features"][:5])
        ['Zone1/FlowSolution/VelocityX', 'Zone1/FlowSolution/VelocityY', ...]
        >>> print(flat_cst)
        {'Zone1/GridCoordinates': array([0., 0.1, 0.2])}
        >>> print(hf_features)
        {'Zone1/FlowSolution/VelocityX': Value(dtype='float32', id=None), ...}
    """

    def values_equal(v1, v2):
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        return v1 == v2

    global_cgns_types = {}
    global_feature_types = {}
    global_constant_leaves = {}
    total_samples = 0

    for split_name, generator in generators.items():
        for sample in tqdm(
            generator(),
            disable=not verbose,
            desc=f"Prepare for HF on split {split_name}",
        ):
            total_samples += 1
            tree = sample.features.data[0.0]
            flat, cgns_types = flatten_cgns_tree(tree)

            for path, value in flat.items():
                # --- CGNS types ---
                if path not in global_cgns_types:
                    global_cgns_types[path] = cgns_types[path]
                elif global_cgns_types[path] != cgns_types[path]:  # pragma: no cover
                    raise ValueError(
                        f"Conflict for path '{path}': {global_cgns_types[path]} vs {cgns_types[path]}"
                    )

                # --- feature types ---
                inferred_feature = infer_hf_features_from_value(value)
                if path not in global_feature_types:
                    global_feature_types[path] = inferred_feature
                else:
                    # sanity check: convert to dict before comparing
                    if repr(global_feature_types[path]) != repr(
                        inferred_feature
                    ):  # pragma: no cover
                        raise ValueError(
                            f"Feature type mismatch for {path}: "
                            f"{global_feature_types[path]} vs {inferred_feature}"
                        )

                if path not in global_constant_leaves:
                    global_constant_leaves[path] = {
                        "value": value,
                        "constant": True,
                        "count": 1,
                    }
                else:
                    entry = global_constant_leaves[path]
                    entry["count"] += 1
                    if entry["constant"] and not values_equal(entry["value"], value):
                        entry["constant"] = False

    # After loop: only keep constants that appeared in all samples
    for path, entry in global_constant_leaves.items():
        if entry["count"] != total_samples:
            entry["constant"] = False

    # Sort dicts by keys
    global_cgns_types = {p: global_cgns_types[p] for p in sorted(global_cgns_types)}
    global_feature_types = {
        p: global_feature_types[p] for p in sorted(global_feature_types)
    }
    global_constant_leaves = {
        p: global_constant_leaves[p] for p in sorted(global_constant_leaves)
    }

    flat_cst = {
        p: e["value"] for p, e in global_constant_leaves.items() if e["constant"]
    }

    cst_features = list(flat_cst.keys())
    var_features = [k for k in global_cgns_types.keys() if k not in cst_features]

    hf_features = Features(
        {k: v for k, v in global_feature_types.items() if k in var_features}
    )

    key_mappings = {}
    key_mappings["variable_features"] = var_features
    key_mappings["constant_features"] = cst_features
    key_mappings["cgns_types"] = global_cgns_types

    return flat_cst, key_mappings, hf_features


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
            tree = sample.features.data[0.0]
            flat, _ = flatten_cgns_tree(tree)
            yield {path: flat.get(path, None) for path in all_features_keys}

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
    enforce_shapes: bool = False,
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
            datasets. Defaults to False.

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
    # Download split.json
    json_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"problem_definitions/{name}/split.json",
        repo_type="dataset",
    )
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

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
    prob_def.set_split(json_data)

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

    data = pb_def.get_split()
    json_str = json.dumps(data)
    json_buffer = io.BytesIO(json_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=json_buffer,
        path_in_repo=f"problem_definitions/{name}/split.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload problem_definitions/{name}/split.json",
    )


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
    pb_def._load_from_dir_(Path(path) / Path("problem_definitions") / Path(name))
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
    pb_def._save_to_dir_(Path(path) / Path("problem_definitions") / Path(name))


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
#     DEPRECATED  HUGGING FACE BRIDGE (binary blobs)
# ------------------------------------------------------------------------------


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
)
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


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
)
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


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
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


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
)
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


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
)
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


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
)
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


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
)
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


@deprecated(
    "will be removed (this hf format will not be not maintained)",
    version="0.1.10",
    removal="1.0.0",
)
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
