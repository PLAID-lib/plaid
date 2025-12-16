# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""CGNS dataset reader module for PLAID.

This module provides functionality for reading and streaming CGNS datasets for the PLAID library.
It includes utilities for loading datasets from local disk or streaming directly from Hugging Face Hub,
with support for selective loading of splits and samples.

Key features:
- Local dataset loading from disk via CGNSDataset class
- Streaming datasets from Hugging Face Hub
- Selective loading of splits and sample IDs
- Integration with PLAID Sample objects
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

import numpy as np
import yaml
from datasets import IterableDataset
from datasets.splits import NamedSplit
from huggingface_hub import hf_hub_download, snapshot_download

from plaid import Sample

logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Classes and functions
# ------------------------------------------------------


class CGNSDataset:
    """CGNS dataset class for local disk access.

    This class represents a CGNS dataset stored on local disk, providing access to individual
    samples and associated metadata. It supports iteration over samples and attribute access
    to extra fields.
    """

    def __init__(self, path: Union[str, Path], **kwargs) -> None:
        """Initialize a :class:`CGNSDataset`.

        Args:
            path: Path to the dataset directory.
            **kwargs: Optional keyword metadata to attach to the dataset instance.
                All provided kwargs are stored in ``self._extra_fields`` and are
                accessible as attributes via ``__getattr__`` / ``__setattr__``.
        """
        self.path = path
        self._extra_fields = dict(kwargs)

        if Path(path).is_dir():
            ids = sorted(
                int(p.name.removeprefix("sample_"))
                for p in path.iterdir()
                if p.is_dir() and p.name.startswith("sample_")
            )
            self._extra_fields["ids"] = np.asarray(ids, dtype=int)

        else:  # pragma: no cover
            raise ValueError("path must be a local directory")

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over all samples in the dataset.

        Yields:
            Sample: A PLAID Sample object for each sample in the dataset.
        """
        for idx in self.ids:
            yield self[idx]

    def __getitem__(self, idx: int) -> Sample:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Sample: A PLAID Sample object.
        """
        assert idx in self.ids
        return Sample(path=self.path / f"sample_{idx:09d}")

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.ids)

    def __getattr__(self, name: str):
        """Get attribute from extra fields.

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
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}'"
        )  # pragma: no cover

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute in extra fields.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        if name in ("path", "_extra_fields"):
            super().__setattr__(name, value)
        else:
            self._extra_fields[name] = value

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns:
            str: String representation.
        """
        return f"<CGNSDataset {repr(self.path)} | extra_fields={self._extra_fields}>"


def sample_generator(
    repo_id: str, split: str, ids: Iterable[int]
) -> Iterator[Sample]:  # pragma: no cover
    """Generate Sample objects from a Hugging Face Hub repository.

    This function downloads individual samples from a CGNS dataset stored on Hugging Face Hub
    and yields PLAID Sample objects. Each sample is downloaded to a temporary directory
    and loaded as a Sample.

    Args:
        repo_id: The Hugging Face repository ID (e.g., 'username/dataset-name').
        split: The dataset split name (e.g., 'train', 'test').
        ids: Iterable of sample IDs to generate.

    Yields:
        Sample: A PLAID Sample object for each requested ID.
    """
    for idx in ids:
        with tempfile.TemporaryDirectory(prefix="plaid_sample_") as temp_folder:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=[f"data/{split}/sample_{idx:09d}/"],
                local_dir=temp_folder,
            )
            sample = Sample(
                path=Path(temp_folder) / "data" / f"{split}" / f"sample_{idx:09d}"
            )
        yield sample


def create_CGNS_iterable_dataset(
    repo_id: str, split: str, ids: Iterable[int]
) -> IterableDataset:  # pragma: no cover
    """Create an iterable dataset from CGNS samples on Hugging Face Hub.

    This function creates a Hugging Face IterableDataset that streams CGNS samples
    from a repository. The dataset can be used for efficient streaming access without
    loading all samples into memory.

    Args:
        repo_id: The Hugging Face repository ID (e.g., 'username/dataset-name').
        split: The dataset split name (e.g., 'train', 'test').
        ids: Iterable of sample IDs to include in the dataset.

    Returns:
        IterableDataset: A Hugging Face IterableDataset for streaming access.
    """
    return IterableDataset.from_generator(
        sample_generator,
        gen_kwargs={"repo_id": repo_id, "split": split, "ids": ids},
        split=NamedSplit(split),
        features=None,
    )


# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------


def init_datasetdict_from_disk(
    path: Union[str, Path],
) -> dict[str, CGNSDataset]:
    """Initialize a dataset dictionary from local disk.

    This function scans a local directory structure and creates CGNSDataset objects
    for each split found in the data directory.

    Args:
        path: Path to the root directory containing the dataset. Should contain a 'data' subdirectory
            with split subdirectories.

    Returns:
        dict[str, CGNSDataset]: Dictionary mapping split names to CGNSDataset objects.
    """
    local_path = Path(path) / "data"
    split_names = [p.name for p in local_path.iterdir() if p.is_dir()]
    return {sn: CGNSDataset(local_path / sn) for sn in split_names}


# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------


def download_datasetdict_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, Iterable[int]]] = None,
    features: Optional[list[str]] = None,  # noqa: ARG001
    overwrite: bool = False,
) -> None:  # pragma: no cover
    """Download a CGNS dataset from Hugging Face Hub to local disk.

    This function downloads selected parts or the entire CGNS dataset from a Hugging Face
    repository to a local directory. Supports selective downloading of specific splits and samples.

    Args:
        repo_id: The Hugging Face repository ID (e.g., 'username/dataset-name').
        local_dir: Local directory path where the dataset will be downloaded.
        split_ids: Optional dictionary mapping split names to iterables of sample IDs to download.
            If None, downloads all splits and samples.
        features: Optional list of features to download (currently unused).
        overwrite: If True, removes existing local directory before downloading.
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

    if split_ids is not None:
        allow_patterns = []
        for split, ids in split_ids.items():
            allow_patterns.extend([f"data/{split}/sample_{i:09d}/*" for i in ids])
    else:
        allow_patterns = ["data/*"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=local_dir,
    )


def init_datasetdict_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, Iterable[int]]] = None,
    features: Optional[list[str]] = None,  # noqa: ARG001
) -> dict[str, IterableDataset]:  # pragma: no cover
    """Initialize streaming datasets from Hugging Face Hub.

    This function creates a dictionary of streaming IterableDataset objects for CGNS data
    stored on Hugging Face Hub. Supports selective streaming of specific splits and samples.

    Args:
        repo_id: The Hugging Face repository ID (e.g., 'username/dataset-name').
        split_ids: Optional dictionary mapping split names to iterables of sample IDs to stream.
            If None, streams all available samples for each split.
        features: Optional list of features to stream (currently unused).

    Returns:
        dict[str, IterableDataset]: Dictionary mapping split names to IterableDataset objects
            for streaming access.
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if hf_endpoint:
        raise RuntimeError("Streaming mode not compatible with private mirror.")

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
        split: create_CGNS_iterable_dataset(repo_id, split, ids)
        for split, ids in selected_ids.items()
    }
