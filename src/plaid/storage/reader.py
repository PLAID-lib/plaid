# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""PLAID storage reader module.

This module provides high-level functions for loading PLAID datasets from local disk or
Hugging Face Hub. It supports multiple storage backends including CGNS, HF Datasets,
and Zarr, providing a unified interface for data access and conversion.

Key features:
- Unified interface for loading datasets across different backends
- Local disk and streaming Hub access
- Automatic backend detection and converter creation
- Sample conversion between storage formats and PLAID objects
"""

from pathlib import Path
from typing import Any, Iterable, Optional, Union

from plaid import Sample
from plaid.storage.cgns.reader import (
    download_datasetdict_from_hub as download_cgns_datasetdict_from_hub,
)
from plaid.storage.cgns.reader import (
    init_datasetdict_from_disk as init_cgns_datasetdict_from_disk,
)
from plaid.storage.cgns.reader import (
    init_datasetdict_streaming_from_hub as init_cgns_datasetdict_streaming_from_hub,
)
from plaid.storage.common.bridge import (
    plaid_to_sample_dict,
    to_plaid_sample,
    to_sample_dict,
)
from plaid.storage.common.reader import (
    load_infos_from_disk,
    load_infos_from_hub,
    load_metadata_from_disk,
    load_metadata_from_hub,
    load_problem_definitions_from_hub,
)
from plaid.storage.common.writer import (
    save_infos_to_disk,
    save_metadata_to_disk,
    save_problem_definitions_to_disk,
)
from plaid.storage.hf_datasets.bridge import (
    sample_to_var_sample_dict as hf_sample_to_var_sample_dict,
)
from plaid.storage.hf_datasets.bridge import (
    to_var_sample_dict as hf_to_var_sample_dict,
)
from plaid.storage.hf_datasets.reader import (
    download_datasetdict_from_hub as download_hf_datasetdict_from_hub,
)
from plaid.storage.hf_datasets.reader import (
    init_datasetdict_from_disk as init_hf_datasetdict_from_disk,
)
from plaid.storage.hf_datasets.reader import (
    init_datasetdict_streaming_from_hub as init_hf_datasetdict_streaming_from_hub,
)
from plaid.storage.zarr.bridge import (
    sample_to_var_sample_dict as zarr_sample_to_var_sample_dict,
)
from plaid.storage.zarr.bridge import (
    to_var_sample_dict as zarr_to_var_sample_dict,
)
from plaid.storage.zarr.reader import (
    download_datasetdict_from_hub as download_zarr_datasetdict_from_hub,
)
from plaid.storage.zarr.reader import (
    init_datasetdict_from_disk as init_zarr_datasetdict_from_disk,
)
from plaid.storage.zarr.reader import (
    init_datasetdict_streaming_from_hub as init_zarr_datasetdict_streaming_from_hub,
)

init_datasetdict_from_disk = {
    "hf_datasets": init_hf_datasetdict_from_disk,
    "cgns": init_cgns_datasetdict_from_disk,
    "zarr": init_zarr_datasetdict_from_disk,
}

download_datasetdict_from_hub = {
    "hf_datasets": download_hf_datasetdict_from_hub,
    "cgns": download_cgns_datasetdict_from_hub,
    "zarr": download_zarr_datasetdict_from_hub,
}

init_datasetdict_streaming_from_hub = {
    "hf_datasets": init_hf_datasetdict_streaming_from_hub,
    "cgns": init_cgns_datasetdict_streaming_from_hub,
    "zarr": init_zarr_datasetdict_streaming_from_hub,
}

to_var_sample_dict = {
    "hf_datasets": hf_to_var_sample_dict,
    "zarr": zarr_to_var_sample_dict,
}

sample_to_var_sample_dict = {
    "hf_datasets": hf_sample_to_var_sample_dict,
    "zarr": zarr_sample_to_var_sample_dict,
}


class Converter:
    """Converter class for transforming samples between storage and PLAID formats.

    This class provides methods to convert samples between backend-specific storage formats
    and PLAID Sample objects. It handles the schema transformations and metadata required
    for proper data conversion.
    """

    def __init__(
        self,
        backend: str,
        flat_cst: Any,
        cgns_types: Any,
        variable_schema: Any,
        constant_schema: Any,
    ) -> None:
        """Initialize a :class:`Converter`.

        Args:
            backend: The storage backend ('hf_datasets', 'zarr', or 'cgns').
            flat_cst: Flattened constants for the dataset.
            cgns_types: CGNS type information.
            variable_schema: Schema for variable fields.
            constant_schema: Schema for constant fields.
        """
        self.backend = backend
        self.flat_cst = flat_cst
        self.cgns_types = cgns_types
        self.variable_schema = variable_schema
        self.constant_schema = constant_schema

    def to_dict(self, dataset: Any, idx: int) -> dict[str, Any]:
        """Convert a dataset sample to dictionary format.

        Args:
            dataset: The dataset object containing the sample.
            idx: Index of the sample to convert.

        Returns:
            dict: Sample data in dictionary format.

        Raises:
            ValueError: If called with CGNS backend.
        """
        if self.backend == "cgns":
            raise ValueError("Converter.to_dict not available for cgns backend")
        var_sample_dict = to_var_sample_dict[self.backend](dataset, idx)
        return to_sample_dict(var_sample_dict, self.flat_cst, self.cgns_types)

    def to_plaid(self, dataset: Any, idx: int) -> Sample:
        """Convert a dataset sample to PLAID Sample object.

        Args:
            dataset: The dataset object containing the sample.
            idx: Index of the sample to convert.

        Returns:
            Sample: A PLAID Sample object.
        """
        if self.backend != "cgns":
            sample_dict = self.to_dict(dataset, idx)
            return to_plaid_sample(sample_dict, self.cgns_types)
        else:
            return dataset[idx]

    def sample_to_dict(self, sample: Sample) -> dict[str, Any]:
        """Convert a PLAID Sample to dictionary format.

        Args:
            sample: The PLAID Sample object to convert.

        Returns:
            dict: Sample data in dictionary format.

        Raises:
            ValueError: If called with CGNS backend.
        """
        if self.backend == "cgns":
            raise ValueError("Converter.sample_to_dict not available for cgns backend")
        var_sample_dict = sample_to_var_sample_dict[self.backend](sample)
        return to_sample_dict(var_sample_dict, self.flat_cst, self.cgns_types)

    def sample_to_plaid(self, sample: Sample) -> Sample:
        """Convert a sample to PLAID format (identity function for most backends).

        Args:
            sample: The sample object to convert.

        Returns:
            Sample: A PLAID Sample object.
        """
        if self.backend != "cgns":
            sample_dict = self.sample_to_dict(sample)
            return to_plaid_sample(sample_dict, self.cgns_types)
        else:
            return sample

    def plaid_to_dict(self, plaid_sample: Sample) -> dict[str, Any]:
        """Convert a PLAID Sample to dictionary format for storage.

        Args:
            plaid_sample: The PLAID Sample object to convert.

        Returns:
            dict: Sample data in dictionary format suitable for storage.
        """
        return plaid_to_sample_dict(
            plaid_sample, self.variable_schema, self.constant_schema
        )

    def __repr__(self) -> str:
        """String representation of the Converter.

        Returns:
            str: String representation including the backend.
        """
        return f"Converter(backend={self.backend})"


def init_from_disk(
    local_dir: Union[Path, str],
) -> tuple[dict[str, Any], dict[str, "Converter"]]:
    """Initialize dataset and converters from local disk.

    This function loads a previously saved PLAID dataset from local disk, automatically
    detecting the backend and creating appropriate converters for sample transformation.

    Args:
        local_dir: Path to the local directory containing the saved dataset.

    Returns:
        tuple: A tuple containing (datasetdict, converterdict) where datasetdict maps
            split names to dataset objects and converterdict maps split names to Converter objects.
    """
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_disk(
        local_dir
    )
    infos = load_infos_from_disk(local_dir)

    backend = infos["storage_backend"]

    datasetdict = init_datasetdict_from_disk[backend](local_dir)

    converterdict = {}
    for split in datasetdict.keys():
        converterdict[split] = Converter(
            backend, flat_cst[str(split)], cgns_types, variable_schema, constant_schema
        )
    return datasetdict, converterdict


def download_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, Iterable[int]]] = None,
    features: Optional[list[str]] = None,
    overwrite: bool = False,
):  # pragma: no cover
    """Download a PLAID dataset from Hugging Face Hub to local disk.

    This function downloads a dataset from Hugging Face Hub, including data, metadata,
    infos, and problem definitions, saving everything to local disk.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name').
        local_dir: Local directory path where the dataset will be downloaded.
        split_ids: Optional dictionary mapping split names to iterables of sample IDs to download.
        features: Optional list of features to download.
        overwrite: If True, overwrites existing local directory.
    """
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_hub(
        repo_id
    )
    infos = load_infos_from_hub(repo_id)
    pb_defs = load_problem_definitions_from_hub(repo_id)

    backend = infos["storage_backend"]

    download_datasetdict_from_hub[backend](
        repo_id, local_dir, split_ids, features, overwrite
    )

    save_metadata_to_disk(
        local_dir, flat_cst, variable_schema, constant_schema, cgns_types
    )
    save_infos_to_disk(local_dir, infos)
    if pb_defs is not None:
        save_problem_definitions_to_disk(local_dir, pb_defs)


def init_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, Iterable[int]]] = None,
    features: Optional[list[str]] = None,
) -> tuple[dict[str, Any], dict[str, "Converter"]]:  # pragma: no cover
    """Initialize streaming datasets from Hugging Face Hub.

    This function creates streaming dataset objects from a Hugging Face Hub repository,
    along with converters for sample transformation.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name').
        split_ids: Optional dictionary mapping split names to iterables of sample IDs to stream.
        features: Optional list of features to stream.

    Returns:
        tuple: A tuple containing (datasetdict, converterdict) where datasetdict maps
            split names to streaming dataset objects and converterdict maps split names to Converter objects.
    """
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_hub(
        repo_id
    )
    infos = load_infos_from_hub(repo_id)

    backend = infos["storage_backend"]

    datasetdict = init_datasetdict_streaming_from_hub[backend](
        repo_id, split_ids, features
    )

    converterdict = {}
    for split in datasetdict.keys():
        converterdict[split] = Converter(
            backend, flat_cst[str(split)], cgns_types, variable_schema, constant_schema
        )

    return datasetdict, converterdict
