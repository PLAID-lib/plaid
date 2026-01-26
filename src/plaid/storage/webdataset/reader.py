"""WebDataset reader module.

This module provides functionality for reading and streaming datasets stored in WebDataset format
for the PLAID library. It includes utilities for loading datasets from local disk or
streaming directly from Hugging Face Hub, with support for selective loading of splits
and features.

Key features:
- Local dataset loading from tar archives
- Streaming datasets from Hugging Face Hub
- Selective loading of splits and features
- WebDatasetWrapper class for convenient data access with indexing support
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import io
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import webdataset as wds
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Classes
# ------------------------------------------------------


class WebDatasetWrapper:
    """A wrapper class for WebDataset providing indexing support.

    This class wraps a WebDataset tar archive and provides indexed access to samples,
    which is required for PLAID's data access patterns. Since WebDataset is designed
    for streaming, this wrapper caches samples on first iteration to enable random access.
    """

    def __init__(
        self, tar_path: Union[str, Path], features: Optional[list[str]] = None
    ) -> None:
        """Initialize a WebDatasetWrapper.

        Args:
            tar_path: Path to the tar file containing the dataset.
            features: Optional list of features to load. If None, all features are loaded.
        """
        self.tar_path = Path(tar_path)
        self.features = features
        self._cache = None
        self._ids = None

    def _load_cache(self) -> None:
        """Load all samples into cache for random access."""
        if self._cache is not None:
            return

        self._cache = []
        self._ids = []

        # Create WebDataset pipeline to read tar
        dataset = wds.WebDataset(str(self.tar_path))

        for sample in dataset:
            # Decode the sample
            decoded_sample = self._decode_sample(sample)

            # Filter features if specified
            if self.features is not None:
                decoded_sample = {
                    k: v for k, v in decoded_sample.items() if k in self.features
                }

            self._cache.append(decoded_sample)

            # Extract sample index from __key__ (e.g., "sample_000000001")
            if "__key__" in sample:
                basename = sample["__key__"]
                if basename.startswith("sample_"):
                    idx = int(basename.split("_")[1])
                    self._ids.append(idx)

        self._ids = np.array(self._ids, dtype=int)

    def _decode_sample(self, sample: dict[str, bytes]) -> dict[str, Any]:
        """Decode a WebDataset sample from bytes to numpy arrays.

        Args:
            sample: Dictionary of extension -> bytes from tar archive.

        Returns:
            dict[str, Any]: Decoded sample with feature paths as keys.
        """
        decoded = {}

        for key, value in sample.items():
            # Skip __key__ metadata
            if key == "__key__":
                continue

            # Handle .npy files
            # Format in dict: "feature__path.npy" -> bytes
            if key.endswith(".npy"):
                # Remove .npy extension to get feature path
                feature_path = key[:-4]
                # Convert __ back to /
                feature_path = feature_path.replace("__", "/")

                # Decode numpy array
                buffer = io.BytesIO(value)
                array = np.load(buffer, allow_pickle=False)
                decoded[feature_path] = array

        return decoded

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over all samples in the dataset.

        Yields:
            dict[str, Any]: Dictionary containing sample data.
        """
        self._load_cache()
        yield from self._cache

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            dict[str, Any]: Dictionary containing sample data.
        """
        self._load_cache()

        # Find the position in cache
        if idx not in self._ids:
            raise IndexError(f"Sample index {idx} not found in dataset")

        position = np.where(self._ids == idx)[0][0]
        return self._cache[position]

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        self._load_cache()
        return len(self._cache)

    @property
    def ids(self) -> np.ndarray:
        """Get array of sample IDs in the dataset.

        Returns:
            np.ndarray: Array of sample indices.
        """
        self._load_cache()
        return self._ids


class WebDatasetDict:
    """A dataset dictionary class for WebDataset format.

    This class provides a dictionary-like interface to access multiple splits of a
    WebDataset, similar to ZarrDataset pattern in PLAID.
    """

    def __init__(
        self,
        path: Union[str, Path],
        split_tar_paths: dict[str, Path],
        features: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Initialize a WebDatasetDict.

        Args:
            path: Path to the dataset root directory.
            split_tar_paths: Dictionary mapping split names to tar file paths.
            features: Optional list of features to load.
            **kwargs: Additional metadata to attach to the dataset.
        """
        self.path = path
        self.split_tar_paths = split_tar_paths
        self.features = features
        self._extra_fields = dict(kwargs)
        self._splits = {}

    def __getitem__(self, split: str) -> WebDatasetWrapper:
        """Get a split by name.

        Args:
            split: Split name.

        Returns:
            WebDatasetWrapper: Wrapper for the split's tar archive.
        """
        if split not in self._splits:
            if split not in self.split_tar_paths:
                raise KeyError(f"Split '{split}' not found in dataset")

            self._splits[split] = WebDatasetWrapper(
                self.split_tar_paths[split], self.features
            )

        return self._splits[split]

    def __len__(self) -> int:
        """Get the number of splits.

        Returns:
            int: Number of splits.
        """
        return len(self.split_tar_paths)

    def __iter__(self) -> Iterator[tuple[str, WebDatasetWrapper]]:
        """Iterate over splits.

        Yields:
            tuple[str, WebDatasetWrapper]: (split_name, dataset_wrapper) pairs.
        """
        for split_name in self.split_tar_paths.keys():
            yield split_name, self[split_name]

    def keys(self):
        """Get split names.

        Returns:
            Iterator of split names.
        """
        return self.split_tar_paths.keys()

    def values(self):
        """Get dataset wrappers.

        Yields:
            WebDatasetWrapper instances.
        """
        for split_name in self.split_tar_paths.keys():
            yield self[split_name]

    def items(self):
        """Get (split_name, dataset_wrapper) pairs.

        Yields:
            tuple[str, WebDatasetWrapper] pairs.
        """
        return self.__iter__()

    def __getattr__(self, name: str) -> Any:
        """Get attribute from extra fields.

        Args:
            name: Attribute name.

        Returns:
            Any: Attribute value.

        Raises:
            AttributeError: If attribute not found.
        """
        if name in self._extra_fields:
            return self._extra_fields[name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in extra fields.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name in ("path", "split_tar_paths", "features", "_extra_fields", "_splits"):
            super().__setattr__(name, value)
        else:
            self._extra_fields[name] = value

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns:
            str: String representation.
        """
        splits = list(self.split_tar_paths.keys())
        return f"<WebDatasetDict path={self.path} splits={splits} | extra_fields={self._extra_fields}>"


# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------


def init_datasetdict_from_disk(
    path: Union[str, Path],
) -> dict[str, WebDatasetWrapper]:
    """Initializes dataset dictionaries from local WebDataset tar files.

    Args:
        path: Path to the local directory containing the dataset.

    Returns:
        dict[str, WebDatasetWrapper]: Dictionary mapping split names to WebDatasetWrapper objects.
    """
    local_path = Path(path) / "data"

    if not local_path.exists():
        raise ValueError(f"Data directory not found: {local_path}")

    # Find all tar files
    tar_files = list(local_path.glob("*.tar"))

    if not tar_files:
        raise ValueError(f"No tar files found in {local_path}")

    # Create split_tar_paths mapping
    split_tar_paths = {f.stem: f for f in tar_files}

    # Create WebDatasetDict
    dataset_dict = WebDatasetDict(path, split_tar_paths)

    # Return as plain dict for compatibility
    return {split: dataset_dict[split] for split in split_tar_paths.keys()}


# ------------------------------------------------------
# Load from Hub
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
        repo_id: The Hugging Face repository ID.
        local_dir: Local directory to download to.
        split_ids: Optional split IDs for selective download (not implemented for WebDataset).
        features: Optional features for selective download (not implemented for WebDataset).
        overwrite: Whether to overwrite existing directory.

    Returns:
        None
    """
    output_folder = Path(local_dir)

    if output_folder.is_dir():
        if overwrite:
            shutil.rmtree(local_dir)
            logger.warning(f"Existing {local_dir} directory has been reset.")
        elif any(output_folder.iterdir()):
            raise ValueError(
                f"directory {local_dir} already exists and is not empty. "
                "Set `overwrite` to True if needed."
            )

    # Note: split_ids and features filtering not implemented for WebDataset
    # These would require streaming and re-packing tar files
    if split_ids is not None:
        logger.warning(
            "split_ids filtering not supported for WebDataset backend, "
            "downloading full dataset"
        )

    if features is not None:
        logger.warning(
            "features filtering not supported for WebDataset backend, "
            "downloading full dataset"
        )

    # Download tar files and metadata
    allow_patterns = ["data/*.tar", "*.yaml", "*.yml", "*.json", "README.md"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=local_dir,
    )


def init_datasetdict_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, list[int]]] = None,
    features: Optional[list[str]] = None,  # noqa: ARG001
) -> dict[str, wds.WebDataset]:  # pragma: no cover
    """Initializes streaming dataset dictionaries from Hugging Face Hub.

    This function creates WebDataset pipelines that stream tar data directly from
    the Hugging Face Hub without downloading files locally.

    Args:
        repo_id: The Hugging Face repository ID.
        split_ids: Optional dictionary mapping split names to sample IDs (not supported).
        features: Optional list of feature names to include.

    Returns:
        dict[str, wds.WebDataset]: Dictionary mapping split names to WebDataset pipelines.
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if hf_endpoint:
        raise RuntimeError("Streaming mode not compatible with private mirror.")

    if split_ids is not None:
        logger.warning(
            "split_ids filtering not supported for WebDataset streaming, "
            "loading all samples"
        )

    # Get list of splits from infos
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename="infos.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        infos = yaml.safe_load(f)

    splits = list(infos.get("num_samples", {}).keys())

    if not splits:
        raise ValueError(f"No splits found in dataset {repo_id}")

    # Create streaming WebDataset for each split
    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data"

    datasets = {}
    for split in splits:
        tar_url = f"{base_url}/{split}.tar"

        # Create WebDataset pipeline
        dataset = wds.WebDataset(tar_url)

        # Add decoding if needed
        # dataset = dataset.decode()

        datasets[split] = dataset

    return datasets
