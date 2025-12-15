# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Zarr dataset reader module.

This module provides functionality for reading and streaming datasets stored in Zarr format
for the PLAID library. It includes utilities for loading datasets from local disk or
streaming directly from Hugging Face Hub, with support for selective loading of splits
and features.

Key features:
- Local dataset loading from disk
- Streaming datasets from Hugging Face Hub
- Selective loading of splits and features
- ZarrDataset class for convenient data access
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

import fsspec
import numpy as np
import yaml
import zarr
from datasets import IterableDataset
from datasets.splits import NamedSplit
from huggingface_hub import hf_hub_download, snapshot_download

from plaid.storage.zarr.bridge import unflatten_zarr_key
from plaid.storage.zarr.writer import _flatten_path

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Classes and functions
# ------------------------------------------------------


class ZarrDataset:
    """A dataset class for accessing Zarr-stored data.

    This class provides a convenient interface for accessing samples stored in Zarr format.
    It wraps a Zarr group and provides dictionary-like access to samples, along with
    additional metadata fields.
    """

    def __init__(self, zarr_group: zarr.Group, **kwargs) -> None:
        """Initialize a :class:`ZarrDataset`.

        Args:
            zarr_group (zarr.Group): The underlying Zarr group containing the data.
            **kwargs: Optional keyword metadata to attach to the dataset instance.
                All provided kwargs are stored in ``self._extra_fields`` and are
                accessible as attributes via ``__getattr__`` / ``__setattr__``.
        """
        self.zarr_group = zarr_group
        self._extra_fields = dict(kwargs)

        ids = sorted(int(name[7:]) for name, _ in zarr_group.groups())
        self._extra_fields["ids"] = np.asarray(ids, dtype=int)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples in the dataset.

        Yields:
            dict[str, Any]: Dictionary containing sample data.
        """
        for idx in self.ids:
            yield self[idx]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            dict[str, Any]: Dictionary containing sample data.
        """
        zarr_sample = self.zarr_group[f"sample_{idx:09d}"]
        return {
            unflatten_zarr_key(path): zarr_sample[path]
            for path in zarr_sample.array_keys()
        }

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.zarr_group)

    def __getattr__(self, name: str) -> Any:
        """Get attribute from extra fields or zarr group.

        Args:
            name: Attribute name.

        Returns:
            Any: Attribute value.

        Raises:
            AttributeError: If attribute not found.
        """
        # fallback to extra fields
        if name in self._extra_fields:
            return self._extra_fields[name]
        # fallback to zarr_group attributes
        if hasattr(self.zarr_group, name):  # pragma: no cover
            return getattr(self.zarr_group, name)
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}'"
        )  # pragma: no cover

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in extra fields.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name in ("zarr_group", "_extra_fields"):
            super().__setattr__(name, value)
        else:
            self._extra_fields[name] = value

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns:
            str: String representation.
        """
        return (
            f"<ZarrDataset {repr(self.zarr_group)} | extra_fields={self._extra_fields}>"
        )


def _zarr_patterns(
    repo_id: str,
    split_ids: Optional[dict[str, list[int]]] = None,
    features: Optional[list[str]] = None,
):  # pragma: no cover
    """Generates allow and ignore patterns for Zarr dataset downloading.

    Args:
        repo_id (str): The Hugging Face repository ID.
        split_ids (Optional[dict[str, list[int]]]): Optional split IDs for selective loading.
        features (Optional[list[str]]): Optional features for selective loading.

    Returns:
        tuple: (allow_patterns, ignore_patterns) for snapshot_download.
    """
    # include only selected sample ids
    if split_ids is not None:
        allow_patterns = []
        for split, ids in split_ids.items():
            allow_patterns.extend([f"data/{split}/zarr.json"])
            allow_patterns.extend([f"data/{split}/sample_{i:09d}/*" for i in ids])
    else:
        allow_patterns = ["data/*"]

    # ignore unwanted features
    ignore_patterns = []
    if features:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="variable_schema.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            variable_schema = yaml.safe_load(f)

        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="constant_schema.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            constant_schema = yaml.safe_load(f)

        all_features = list(variable_schema.keys()) + list(constant_schema.keys())
        ignored_features = [f for f in all_features if f not in features]

        ignore_patterns += [
            f"data/*/{_flatten_path(feat)}/*" for feat in ignored_features
        ]

    return allow_patterns, ignore_patterns


def sample_generator(
    repo_id: str, split: str, ids: Iterable[int], selected_features: list[str]
) -> Iterator[dict[str, Any]]:  # pragma: no cover
    """Generates samples from a Zarr dataset on Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face repository ID.
        split (str): The dataset split name.
        ids (Iterable[int]): Iterable of sample IDs to generate.
        selected_features (list[str]): List of features to include.

    Yields:
        dict: Dictionary mapping feature names to Zarr arrays.
    """
    base_url = (
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{split}/sample_"
    )
    for idx in ids:
        sample = {}
        for feat in selected_features:
            flatten_feat = _flatten_path(feat)
            mapper = fsspec.get_mapper(base_url + f"{idx:09d}/{flatten_feat}")
            sample[feat] = zarr.open(mapper, mode="r")

        yield sample


def create_zarr_iterable_dataset(
    repo_id: str, split: str, ids: Iterable[int], selected_features: list[str]
) -> IterableDataset:  # pragma: no cover
    """Creates an IterableDataset from Zarr data on Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face repository ID.
        split (str): The dataset split name.
        ids (Iterable[int]): Iterable of sample IDs.
        selected_features (list[str]): List of features to include.

    Returns:
        IterableDataset: An iterable dataset for streaming access.
    """

    def wrapped_gen():
        yield from sample_generator(repo_id, split, ids, selected_features)

    return IterableDataset.from_generator(
        wrapped_gen,
        split=NamedSplit(split),
        features=None,
    )


# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------


def init_datasetdict_from_disk(
    path: Union[str, Path],
) -> dict[str, ZarrDataset]:
    """Initializes dataset dictionaries from local Zarr files.

    Args:
        path (Union[str, Path]): Path to the local directory containing the dataset.

    Returns:
        dict[str, ZarrDataset]: Dictionary mapping split names to ZarrDataset objects.
    """
    local_path = Path(path) / "data"
    split_names = [p.name for p in local_path.iterdir() if p.is_dir()]
    return {
        sn: ZarrDataset(zarr.open(zarr.storage.LocalStore(local_path / sn), mode="r"))
        for sn in split_names
    }


# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------


def download_datasetdict_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, list[int]]] = None,
    features: Optional[list[str]] = None,
    overwrite: bool = False,
) -> None:  # pragma: no cover
    """Downloads dataset from Hugging Face Hub to local directory.

    Args:
        repo_id (str): The Hugging Face repository ID.
        local_dir (Union[str, Path]): Local directory to download to.
        split_ids (Optional[dict[str, list[int]]]): Optional split IDs for selective download.
        features (Optional[list[str]]): Optional features for selective download.
        overwrite (bool): Whether to overwrite existing directory.

    Returns:
        None
    """
    output_folder = Path(local_dir)

    if output_folder.is_dir():
        if overwrite:
            shutil.rmtree(local_dir)
            logger.warning(f"Existing {local_dir} directory has been reset.")
        elif any(local_dir.iterdir()):
            raise ValueError(
                f"directory {local_dir} already exists and is not empty. Set `overwrite` to True if needed."
            )

    allow_patterns, ignore_patterns = _zarr_patterns(repo_id, split_ids, features)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir=local_dir,
    )


def init_datasetdict_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, list[int]]] = None,
    features: Optional[list[str]] = None,
) -> dict[str, IterableDataset]:  # pragma: no cover
    """Initializes streaming dataset dictionaries from Hugging Face Hub.

    This function creates iterable datasets that stream Zarr data directly from
    the Hugging Face Hub without downloading files locally. It supports selective
    loading of specific splits and features for memory-efficient data access.
    Note that streaming mode is not compatible with private Hugging Face mirrors.

    Args:
        repo_id (str): The Hugging Face repository ID (e.g., "username/dataset_name").
        split_ids (Optional[dict[str, list[int]]]): Optional dictionary mapping split names
            to lists of sample IDs to include. If None, all samples from all splits
            are included.
        features (Optional[list[str]]): Optional list of feature names to include.
            If None, all features from the variable schema are included.

    Returns:
        dict[str, IterableDataset]: Dictionary mapping split names to IterableDataset
            objects for streaming data access.
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if hf_endpoint:
        raise RuntimeError("Streaming mode not compatible with private mirror.")

    if features is not None:
        selected_features = features
    else:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="variable_schema.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            variable_schema = yaml.safe_load(f)
        selected_features = list(variable_schema.keys())

    if split_ids is not None:
        selected_ids = split_ids
    else:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="infos.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            infos = yaml.safe_load(f)
        selected_ids = {
            split: range(n_samples) for split, n_samples in infos["num_samples"].items()
        }

    return {
        split: create_zarr_iterable_dataset(repo_id, split, ids, selected_features)
        for split, ids in selected_ids.items()
    }
