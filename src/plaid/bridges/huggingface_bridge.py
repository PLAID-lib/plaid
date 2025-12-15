"""Hugging Face bridge for PLAID datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
import os
import pickle
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pyarrow as pa
import yaml
from tqdm import tqdm

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

# ------------------------------------------------------------------------------
# imports for the saved functions at the end of the file
import hashlib
import io
import logging
import multiprocessing as mp
import traceback
from functools import partial
from queue import Empty
from typing import Callable

import datasets
from datasets import Features, Sequence, Value, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pydantic import ValidationError

from plaid import Dataset, ProblemDefinition, Sample
from plaid.containers.features import SampleFeatures
from plaid.storage.common.tree_handling import (
    flatten_cgns_tree,
    unflatten_cgns_tree,
)
from plaid.types import IndexType

# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
pa.set_memory_pool(pa.system_memory_pool())


# ------------------------------------------------------------------------------
#     HUGGING FACE BRIDGE (with tree flattening and pyarrow tables)
# ------------------------------------------------------------------------------


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

    Note:
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
                    elif isinstance(value, pa.StringArray):  # pragma: no cover
                        row[name] = value.to_numpy(zero_copy_only=False)
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

    Note:
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
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    # Download problem_infos.yaml
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"problem_definitions/{name}",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    prob_def = ProblemDefinition()
    prob_def._initialize_from_problem_infos_dict(yaml_data)

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


def huggingface_dataset_to_plaid(
    ds: datasets.Dataset,
    ids: Optional[list[int]] = None,
    processes_number: int = 1,
    large_dataset: bool = False,
    verbose: bool = True,
) -> tuple[Union[Dataset, ProblemDefinition], ProblemDefinition]:
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

    dataset.set_infos(infos, warn=False)

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
            logger.error(f"Could not retrieve key:'{key}' from description")
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


#########################################################################################
#################################### SAVED FUNCTIONS ####################################
#########################################################################################
################### kept temporarily in case of lost functionalities ####################


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

    Note:
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
        elif np.issubdtype(dtype, np.dtype("|S1")) or np.issubdtype(
            dtype, np.dtype("<U10")
        ):  # pragma: no cover
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

    Note:
        - Byte-array encoded strings (dtype ``"|S1"``) are handled by reassembling and
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


def _hash_value(value):
    """Compute a hash for a value (np.ndarray or basic types)."""
    if isinstance(value, np.ndarray):
        return hashlib.md5(value.view(np.uint8)).hexdigest()
    return hashlib.md5(str(value).encode("utf-8")).hexdigest()


def process_shard(
    generator_fn: Callable[..., Any],
    progress: Any,
    n_proc: int,
    shard_ids: Optional[list[IndexType]] = None,
) -> tuple[
    set[str],
    dict[str, str],
    dict[str, Union[Value, Sequence]],
    dict[str, dict[str, Union[str, bool, int]]],
    int,
]:
    """Process a single shard of sample ids and collect per-shard metadata.

    This function drives a shard-level pass over samples produced by `generator_fn`.
    For each sample it:
    - flattens the sample into Hugging Face friendly arrays (build_hf_sample),
    - collects observed flattened paths,
    - aggregates CGNS type metadata,
    - infers Hugging Face feature types for each path,
    - detects per-path constants using a content hash,
    - updates progress (either a multiprocessing.Queue or a tqdm progress bar).

    Args:
        shard_ids (list[IndexType]): Sequence of sample ids (a single shard) to process.
        generator_fn (Callable): Generator function accepting a list of shard id sequences
            and yielding Sample objects for those ids.
        progress (Any): Progress reporter; either a multiprocessing.Queue (for parallel
            execution) or a tqdm progress bar object (for sequential execution).
        n_proc (int): Number of worker processes used by the caller (used to decide
            how to report progress).

    Returns:
        tuple:
            - split_all_paths (set[str]): Set of all flattened feature paths observed in the shard.
            - shard_global_cgns_types (dict[str, str]): Mapping path -> CGNS node type observed in the shard.
            - shard_global_feature_types (dict[str, Union[Value, Sequence]]): Inferred HF feature types per path.
            - split_constant_leaves (dict[str, dict]): Per-path metadata for constant detection. Each entry
              is a dict with keys "hash" (str), "constant" (bool) and "count" (int).
            - n_samples_processed (int): Number of samples processed in this shard.

    Raises:
        ValueError: If inconsistent feature types are detected for the same path within the shard.
    """
    split_constant_leaves = {}
    split_all_paths = set()
    shard_global_cgns_types = {}
    shard_global_feature_types = {}

    if shard_ids is not None:
        generator = generator_fn([shard_ids])
    else:
        generator = generator_fn()

    n_samples = 0
    for sample in generator:
        hf_sample, all_paths, sample_cgns_types = build_hf_sample(sample)

        split_all_paths.update(hf_sample.keys())
        shard_global_cgns_types.update(sample_cgns_types)

        # Feature type inference
        for path in all_paths:
            value = hf_sample[path]
            if value is None:
                continue
            inferred = infer_hf_features_from_value(value)
            if path not in shard_global_feature_types:
                shard_global_feature_types[path] = inferred
            elif repr(shard_global_feature_types[path]) != repr(inferred):
                raise ValueError(
                    f"Feature type mismatch for {path} in shard"
                )  # pragma: no cover

        # Constant detection using **hash only**
        for path, value in hf_sample.items():
            h = _hash_value(value)
            if path not in split_constant_leaves:
                split_constant_leaves[path] = {"hashes": {h}, "count": 1}
            else:
                entry = split_constant_leaves[path]
                entry["hashes"].add(h)
                entry["count"] += 1

        # Progress
        if n_proc > 1:
            progress.put(1)  # pragma: no cover
        else:
            progress.update(1)

        n_samples += 1

    return (
        split_all_paths,
        shard_global_cgns_types,
        shard_global_feature_types,
        split_constant_leaves,
        n_samples,
    )


def _process_shard_debug(
    generator_fn, progress_queue, n_proc, shard_ids
):  # pragma: no cover
    try:
        return process_shard(generator_fn, progress_queue, n_proc, shard_ids)
    except Exception as e:
        print(f"Exception in worker for shards {shard_ids}: {e}", file=sys.stderr)
        traceback.print_exc()
        raise  # re-raise to propagate to main process


def preprocess_splits(
    generators: dict[str, Callable],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    processes_number: int = 1,
    verbose: bool = True,
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, Any]],
    dict[str, set[str]],
    dict[str, str],
    dict[str, Union[Value, Sequence]],
]:
    """Pre-process dataset splits: inspect samples to infer features, constants and CGNS metadata.

    This function iterates over the provided split generators (optionally in parallel),
    flattens each PLAID sample into Hugging Face friendly arrays, detects constant
    CGNS leaves (features identical across all samples in a split), infers global
    Hugging Face feature types, and aggregates CGNS type metadata.

    The work is sharded per-split and each shard is processed by `process_shard`.
    In parallel mode, progress is updated via a multiprocessing.Queue; otherwise a
    tqdm progress bar is used.

    Args:
        generators (dict[str, Callable]):
            Mapping from split name to a generator function. Each generator must
            accept a single argument (a sequence of shard ids) and yield PLAID samples.
        gen_kwargs (dict[str, dict[str, list[IndexType]]]):
            Per-split kwargs used to drive generator invocation (e.g. {"train": {"shards_ids": [...]}}).
        processes_number (int, optional):
            Number of worker processes to use for shard-level parallelism. Defaults to 1.
        verbose (bool, optional):
            If True, displays progress bars. Defaults to True.

    Returns:
        tuple:
            - split_all_paths (dict[str, set[str]]):
                For each split, the set of all observed flattened feature paths (including "_times" keys).
            - split_flat_cst (dict[str, dict[str, Any]]):
                For each split, a mapping of constant feature path -> value (constant parts of the tree).
            - split_var_path (dict[str, set[str]]):
                For each split, the set of variable feature paths (non-constant).
            - global_cgns_types (dict[str, str]):
                Aggregated mapping from flattened path -> CGNS node type.
            - global_feature_types (dict[str, Union[Value, Sequence]]):
                Aggregated inferred Hugging Face feature types for each variable path.

    Raises:
        ValueError: If inconsistent feature types or CGNS types are detected across shards/splits.
    """
    global_cgns_types = {}
    global_feature_types = {}
    split_flat_cst = {}
    split_var_path = {}
    split_all_paths = {}

    gen_kwargs_ = gen_kwargs or {split_name: {} for split_name in generators.keys()}

    for split_name, generator_fn in generators.items():
        shards_ids_list = gen_kwargs_[split_name].get("shards_ids", [None])
        n_proc = max(1, processes_number or len(shards_ids_list))

        shards_data = []

        if n_proc == 1:
            with tqdm(
                disable=not verbose,
                desc=f"Pre-process split {split_name}",
            ) as pbar:
                for shard_ids in shards_ids_list:
                    shards_data.append(
                        process_shard(generator_fn, pbar, n_proc=1, shard_ids=shard_ids)
                    )

        else:  # pragma: no cover
            # Parallel execution
            manager = mp.Manager()
            progress_queue = manager.Queue()
            shards_data = []

            try:
                with mp.Pool(n_proc) as pool:
                    results = [
                        pool.apply_async(
                            _process_shard_debug,
                            args=(generator_fn, progress_queue, n_proc, shard_ids),
                        )
                        for shard_ids in shards_ids_list
                    ]

                    total_samples = sum(len(shard) for shard in shards_ids_list)
                    completed = 0

                    with tqdm(
                        total=total_samples,
                        disable=not verbose,
                        desc=f"Pre-process split {split_name}",
                    ) as pbar:
                        while completed < total_samples:
                            try:
                                increment = progress_queue.get(timeout=0.5)
                                pbar.update(increment)
                                completed += increment
                            except Empty:
                                # Check for any crashed workers
                                for r in results:
                                    if r.ready():
                                        try:
                                            r.get(
                                                timeout=0.1
                                            )  # will raise worker exception if any
                                        except Exception as e:
                                            raise RuntimeError(f"Worker crashed: {e}")

                    # Collect all results
                    for r in results:
                        shards_data.append(r.get())

            finally:
                manager.shutdown()

        # Merge shard results
        split_all_paths[split_name] = set()
        split_constant_hashes = {}
        n_samples_total = 0

        for (
            all_paths,
            shard_cgns,
            shard_features,
            shard_constants,
            n_samples,
        ) in shards_data:
            split_all_paths[split_name].update(all_paths)
            global_cgns_types.update(shard_cgns)

            for path, inferred in shard_features.items():
                if path not in global_feature_types:
                    global_feature_types[path] = inferred
                elif repr(global_feature_types[path]) != repr(inferred):
                    raise ValueError(  # pragma: no cover
                        f"Feature type mismatch for {path} in split {split_name}"
                    )

            for path, entry in shard_constants.items():
                if path not in split_constant_hashes:
                    split_constant_hashes[path] = entry
                else:
                    existing = split_constant_hashes[path]
                    existing["hashes"].update(entry["hashes"])
                    existing["count"] += entry["count"]

            n_samples_total += n_samples

        # Determine truly constant paths (same hash across all samples)
        constant_paths = [
            p
            for p, entry in split_constant_hashes.items()
            if len(entry["hashes"]) == 1 and entry["count"] == n_samples_total
        ]

        # Retrieve **values** only for constant paths from first sample
        if gen_kwargs:
            first_sample = next(generator_fn([shards_ids_list[0]]))
        else:
            first_sample = next(generator_fn())
        hf_sample, _, _ = build_hf_sample(first_sample)

        split_flat_cst[split_name] = {p: hf_sample[p] for p in sorted(constant_paths)}
        split_var_path[split_name] = {
            p
            for p in split_all_paths[split_name]
            if p not in split_flat_cst[split_name]
        }

    global_feature_types = {
        p: global_feature_types[p] for p in sorted(global_feature_types)
    }

    return (
        split_all_paths,
        split_flat_cst,
        split_var_path,
        global_cgns_types,
        global_feature_types,
    )


def _generator_prepare_for_huggingface(
    generators: dict[str, Callable],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    processes_number: int = 1,
    verbose: bool = True,
):
    (
        split_all_paths,
        split_flat_cst,
        split_var_path,
        global_cgns_types,
        global_feature_types,
    ) = preprocess_splits(generators, gen_kwargs, processes_number, verbose)

    # --- build HF features ---
    var_features = sorted(list(set().union(*split_var_path.values())))
    if len(var_features) == 0:  # pragma: no cover
        raise ValueError(
            "no variable feature found, is your dataset variable through samples?"
        )

    for split_name in split_flat_cst.keys():
        for path in var_features:
            if not path.endswith("_times") and path not in split_all_paths[split_name]:
                split_flat_cst[split_name][path + "_times"] = None  # pragma: no cover
            if path in split_flat_cst[split_name]:
                split_flat_cst[split_name].pop(path)  # pragma: no cover

    cst_features = {
        split_name: sorted(list(cst.keys()))
        for split_name, cst in split_flat_cst.items()
    }
    first_split, first_value = next(iter(cst_features.items()), (None, None))
    for split, value in cst_features.items():
        assert value == first_value, (
            f"cst_features differ for split '{split}' (vs '{first_split}')"
        )
    cst_features = first_value

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


# # -------------------------------------------
# # --------- Sequential version
# def _generator_prepare_for_huggingface(
#     generators: dict[str, Callable],
#     gen_kwargs: dict,
#     processes_number: int = 1,
#     verbose: bool = True,
# ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], Features]:
#     """Inspect PLAID dataset generators and infer Hugging Face feature schema.

#     Iterates over all samples in all provided split generators to:
#       1. Flatten each CGNS tree into a dictionary of paths → values.
#       2. Infer Hugging Face `Features` types for all variable leaves.
#       3. Detect constant leaves (values that never change across all samples).
#       4. Collect global CGNS type metadata.

#     Args:
#         generators (dict[str, Callable]):
#             Mapping from split names to callables returning sample generators.
#             Each sample must have `sample.features.data[0.0]` compatible with `flatten_cgns_tree`.
#         gen_kwargs (dict, optional, default=None):
#             Optional mapping from split names to dictionaries of keyword arguments
#             to be passed to each generator function, used for parallelization.
#         processes_number (int, optional): Number of parallel processes to use.
#         verbose (bool, optional): If True, displays progress bars while processing splits.

#     Returns:
#         tuple:
#             - flat_cst (dict[str, Any]): Mapping from feature path to constant values detected across all splits.
#             - key_mappings (dict[str, Any]): Metadata dictionary with:
#                 - "variable_features" (list[str]): paths of non-constant features.
#                 - "constant_features" (list[str]): paths of constant features.
#                 - "cgns_types" (dict[str, Any]): CGNS type information for all paths.
#             - hf_features (datasets.Features): Hugging Face feature specification for variable features.

#     Raises:
#         ValueError: If inconsistent CGNS types or feature types are found for the same path.
#     """
#     processes_number

#     def values_equal(v1, v2):
#         if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
#             return np.array_equal(v1, v2)
#         return v1 == v2

#     global_cgns_types = {}
#     global_feature_types = {}

#     split_flat_cst = {}
#     split_var_path = {}
#     split_all_paths = {}

#     # ---- Single pass over all splits and samples ----
#     for split_name, generator in generators.items():
#         split_constant_leaves = {}

#         split_all_paths[split_name] = set()

#         n_samples = 0
#         for sample in tqdm(
#             generator(**gen_kwargs[split_name]),
#             disable=not verbose,
#             desc=f"Pre-process split {split_name}",
#         ):
#             # --- Build Hugging Face–compatible sample ---
#             hf_sample, all_paths, sample_cgns_types = build_hf_sample(sample)

#             split_all_paths[split_name].update(hf_sample.keys())
#             # split_all_paths[split_name].update(all_paths)
#             global_cgns_types.update(sample_cgns_types)

#             # --- Infer global HF feature types ---
#             for path in all_paths:
#                 value = hf_sample[path]
#                 if value is None:
#                     continue

#                 # if isinstance(value, np.ndarray) and value.dtype.type is np.str_:
#                 #     inferred = Value("string")
#                 # else:
#                 #     inferred = infer_hf_features_from_value(value)

#                 inferred = infer_hf_features_from_value(value)

#                 if path not in global_feature_types:
#                     global_feature_types[path] = inferred
#                 elif repr(global_feature_types[path]) != repr(inferred):
#                     raise ValueError(  # pragma: no cover
#                         f"Feature type mismatch for {path} in split {split_name}"
#                     )

#             # --- Update per-split constant detection ---
#             for path, value in hf_sample.items():
#                 if path not in split_constant_leaves:
#                     split_constant_leaves[path] = {
#                         "value": value,
#                         "constant": True,
#                         "count": 1,
#                     }
#                 else:
#                     entry = split_constant_leaves[path]
#                     entry["count"] += 1
#                     if entry["constant"] and not values_equal(entry["value"], value):
#                         entry["constant"] = False

#             n_samples += 1

#         # --- Record per-split constants ---
#         for p, e in split_constant_leaves.items():
#             if e["count"] < n_samples:
#                 split_constant_leaves[p]["constant"] = False

#         split_flat_cst[split_name] = dict(
#             sorted(
#                 (
#                     (p, e["value"])
#                     for p, e in split_constant_leaves.items()
#                     if e["constant"]
#                 ),
#                 key=lambda x: x[0],
#             )
#         )

#         split_var_path[split_name] = {
#             p
#             for p in split_all_paths[split_name]
#             if p not in split_flat_cst[split_name]
#         }

#     global_feature_types = {
#         p: global_feature_types[p] for p in sorted(global_feature_types)
#     }
#     var_features = sorted(list(set().union(*split_var_path.values())))

#     if len(var_features) == 0:
#         raise ValueError(  # pragma: no cover
#             "no variable feature found, is your dataset variable through samples?"
#         )

#     # ---------------------------------------------------
#     # for test-like splits, some var_features are all None (e.g.: outputs): need to add '_times' counterparts to corresponding constant trees
#     for split_name in split_flat_cst.keys():
#         for path in var_features:
#             if not path.endswith("_times") and path not in split_all_paths[split_name]:
#                 split_flat_cst[split_name][path + "_times"] = None  # pragma: no cover
#             if (
#                 path in split_flat_cst[split_name]
#             ):  # remove for flat_cst the path that will be forcely included in the arrow tables
#                 split_flat_cst[split_name].pop(path)  # pragma: no cover

#     # ---- Constant features sanity check
#     cst_features = {
#         split_name: sorted(list(cst.keys()))
#         for split_name, cst in split_flat_cst.items()
#     }

#     first_split, first_value = next(iter(cst_features.items()), (None, None))
#     for split, value in cst_features.items():
#         assert value == first_value, (
#             f"cst_features differ for split '{split}' (vs '{first_split}'): something went wrong in _generator_prepare_for_huggingface."
#         )

#     cst_features = first_value

#     # ---- Build global HF Features (only variable) ----
#     hf_features_map = {}
#     for k in var_features:
#         if k.endswith("_times"):
#             hf_features_map[k] = Sequence(Value("float64"))  # pragma: no cover
#         else:
#             hf_features_map[k] = global_feature_types[k]

#     hf_features = Features(hf_features_map)

#     var_features = [path for path in var_features if not path.endswith("_times")]
#     cst_features = [path for path in cst_features if not path.endswith("_times")]

#     key_mappings = {
#         "variable_features": var_features,
#         "constant_features": cst_features,
#         "cgns_types": global_cgns_types,
#     }

#     return split_flat_cst, key_mappings, hf_features


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

    # gen_kwargs = {
    #     split_name: {"shards_ids": [ids]} for split_name, ids in main_splits.items()
    # }

    return plaid_generator_to_huggingface_datasetdict(
        generators,
        processes_number=processes_number,
        writer_batch_size=writer_batch_size,
        verbose=verbose,
    )


def plaid_generator_to_huggingface_datasetdict(
    generators: dict[str, Callable],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
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
        gen_kwargs (dict, optional, default=None):
            Optional mapping from split names to dictionaries of keyword arguments
            to be passed to each generator function, used for parallelization.
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
        generators, gen_kwargs, processes_number, verbose
    )

    all_features_keys = list(hf_features.keys())

    def generator_fn(gen_func, all_features_keys, **kwargs):
        for sample in gen_func(**kwargs):
            hf_sample, _, _ = build_hf_sample(sample)
            yield {path: hf_sample.get(path, None) for path in all_features_keys}

    _dict = {}
    for split_name, gen_func in generators.items():
        gen = partial(generator_fn, all_features_keys=all_features_keys)
        gen_kwargs_ = gen_kwargs or {split_name: {} for split_name in generators.keys()}
        _dict[split_name] = datasets.Dataset.from_generator(
            generator=gen,
            gen_kwargs={"gen_func": gen_func, **gen_kwargs_[split_name]},
            features=hf_features,
            num_proc=processes_number,
            writer_batch_size=writer_batch_size,
            split=datasets.splits.NamedSplit(split_name),
        )

    return datasets.DatasetDict(_dict), flat_cst, key_mappings


def _compute_num_shards(hf_dataset_dict: datasets.DatasetDict) -> dict[str, int]:
    target_shard_size_mb = 500

    num_shards = {}
    for split_name, ds in hf_dataset_dict.items():
        n_samples = len(ds)
        assert n_samples > 0, f"split {split_name} has no sample"

        dataset_size_bytes = ds.data.nbytes
        target_shard_size_bytes = target_shard_size_mb * 1024 * 1024

        n_shards = max(
            1,
            (dataset_size_bytes + target_shard_size_bytes - 1)
            // target_shard_size_bytes,
        )
        num_shards[split_name] = min(n_samples, int(n_shards))
    return num_shards


def push_dataset_dict_to_hub(
    repo_id: str, hf_dataset_dict: datasets.DatasetDict, **kwargs
) -> None:  # pragma: no cover (not tested in unit tests)
    """Push a Hugging Face `DatasetDict` to the Hugging Face Hub.

    This is a thin wrapper around `datasets.DatasetDict.push_to_hub`, allowing
    you to upload a dataset dictionary (with one or more splits such as
    `"train"`, `"validation"`, `"test"`) to the Hugging Face Hub.

    Note:
        The function automatically handles sharding of the dataset by setting `num_shards`
        for each split. For each split, the number of shards is set to the minimum between
        the number of samples in that split and such that shards are targetted to approx. 500 MB.
        This ensures efficient chunking while preventing excessive fragmentation. Empty splits
        will raise an assertion error.

    Args:
        repo_id (str):
            The repository ID on the Hugging Face Hub
            (e.g. `"username/dataset_name"`).
        hf_dataset_dict (datasets.DatasetDict):
            The Hugging Face dataset dictionary to push.
        **kwargs:
            Keyword arguments forwarded to
            [`DatasetDict.push_to_hub`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub).

    Returns:
        None
    """
    num_shards = _compute_num_shards(hf_dataset_dict)
    num_proc = kwargs.get("num_proc", None)
    if num_proc is not None:  # pragma: no cover
        min_num_shards = min(num_shards.values())
        if min_num_shards < num_proc:
            logger.warning(
                f"num_proc chaged from {num_proc} to 1 to safely adapt for num_shards={num_shards}"
            )
            num_proc = 1
        del kwargs["num_proc"]

    hf_dataset_dict.push_to_hub(
        repo_id, num_shards=num_shards, num_proc=num_proc, **kwargs
    )


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

    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo=f"problem_definitions/{name}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload problem_definitions/{name}",
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

    Note:
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


def save_dataset_dict_to_disk(
    path: Union[str, Path], hf_dataset_dict: datasets.DatasetDict, **kwargs
) -> None:
    """Save a Hugging Face DatasetDict to disk.

    This function serializes the provided DatasetDict and writes it to the specified
    directory, preserving its features, splits, and data for later loading.

    Args:
        path (Union[str, Path]): Directory path where the DatasetDict will be saved.
        hf_dataset_dict (datasets.DatasetDict): The Hugging Face DatasetDict to save.
        **kwargs:
            Keyword arguments forwarded to
            [`DatasetDict.save_to_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.save_to_disk).

    Returns:
        None
    """
    num_shards = _compute_num_shards(hf_dataset_dict)
    num_proc = kwargs.get("num_proc", None)
    if num_proc is not None:  # pragma: no cover
        min_num_shards = min(num_shards.values())
        if min_num_shards < num_proc:
            logger.warning(
                f"num_proc chaged from {num_proc} to 1 to safely adapt for num_shards={num_shards}"
            )
            num_proc = 1
        del kwargs["num_proc"]

    hf_dataset_dict.save_to_disk(
        str(path), num_shards=num_shards, num_proc=num_proc, **kwargs
    )


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
    pb_def.save_to_file(Path(path) / Path("problem_definitions") / Path(name))


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

    with open(Path(path) / "tree_constant_part.pkl", "wb") as f:
        pickle.dump(flat_cst, f)

    with open(Path(path) / "key_mappings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(key_mappings, f, sort_keys=False)


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

    Note:
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


def update_dataset_card(
    dataset_card: str,
    infos: Optional[dict[str, dict[str, str]]] = None,
    pretty_name: Optional[str] = None,
    dataset_long_description: Optional[str] = None,
    illustration_urls: Optional[list[str]] = None,
    arxiv_paper_urls: Optional[list[str]] = None,
) -> str:
    r"""Update a dataset card with PLAID-specific metadata and documentation.

    Args:
        dataset_card (str): The original dataset card content to update.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset information
            with "legal" and "data_production" sections. Defaults to None.
        pretty_name (str, optional): A human-readable name for the dataset. Defaults to None.
        dataset_long_description (str, optional): Detailed description of the dataset's content,
            purpose, and characteristics. Defaults to None.
        illustration_urls (list[str], optional): List of URLs to images illustrating the dataset.
            Defaults to None.
        arxiv_paper_urls (list[str], optional): List of URLs to related arXiv papers.
            Defaults to None.

    Returns:
        str: The updated dataset card content as a string.
    """
    lines = dataset_card.splitlines()
    lines = [s for s in lines if not s.startswith("license")]

    indices = [i for i, line in enumerate(lines) if line.strip() == "---"]

    assert len(indices) >= 2, (
        "Cannot find two instances of '---', you should try to update a correct dataset_card."
    )
    lines = lines[: indices[1] + 1]

    count = 1
    lines.insert(count, f"license: {infos['legal']['license']}")
    count += 1
    lines.insert(count, "task_categories:")
    count += 1
    lines.insert(count, "- graph-ml")
    count += 1
    if pretty_name:
        lines.insert(count, f"pretty_name: {pretty_name}")
        count += 1
    lines.insert(count, "tags:")
    count += 1
    lines.insert(count, "- physics learning")
    count += 1
    lines.insert(count, "- geometry learning")
    count += 1

    str__ = "\n".join(lines) + "\n"

    if illustration_urls:
        str__ += "<p align='center'>\n"
        for url in illustration_urls:
            str__ += f"<img src='{url}' alt='{url}' width='1000'/>\n"
        str__ += "</p>\n\n"

    if infos:
        str__ += (
            f"```yaml\n{yaml.dump(infos, sort_keys=False, allow_unicode=True)}\n```"
        )

    str__ += """
Example of commands:
```python
from datasets import load_dataset
from plaid.bridges import huggingface_bridge

repo_id = "chanel/dataset"
pb_def_name = "pb_def_name" #`pb_def_name` is to choose from the repo `problem_definitions` folder

# Load the dataset
hf_datasetdict = load_dataset(repo_id)

# Load addition required data
flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, pb_def_name)

# Efficient reconstruction of plaid samples
for split_name, hf_dataset in hf_datasetdict.items():
    for i in range(len(hf_dataset)):
        sample = huggingface_bridge.to_plaid_sample(
            hf_dataset,
            i,
            flat_cst[split_name],
            key_mappings["cgns_types"],
        )

# Extract input and output features from samples:
for t in sample.get_all_mesh_times():
    for path in pb_def.get_in_features_identifiers():
        sample.get_feature_by_path(path=path, time=t)
    for path in pb_def.get_out_features_identifiers():
        sample.get_feature_by_path(path=path, time=t)
```
"""
    str__ += "This dataset was generated in [PLAID](https://plaid-lib.readthedocs.io/), we refer to this documentation for additional details on how to extract data from `sample` objects.\n"

    if dataset_long_description:
        str__ += f"""
### Dataset Description
{dataset_long_description}
"""

    if arxiv_paper_urls:
        str__ += """
### Dataset Sources

- **Papers:**
"""
        for url in arxiv_paper_urls:
            str__ += f"   - [arxiv]({url})\n"

    return str__
