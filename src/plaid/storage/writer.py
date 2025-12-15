# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""PLAID storage writer module.

This module provides high-level functions for saving PLAID datasets to local disk and pushing
them to Hugging Face Hub. It supports multiple storage backends including CGNS, HF Datasets,
and Zarr, abstracting the backend-specific implementations.

Key features:
- Unified interface for saving datasets across different backends
- Automatic preprocessing and schema extraction
- Metadata and problem definition handling
- Hub integration with dataset cards and metadata
"""

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

from packaging.version import Version

import plaid
from plaid import ProblemDefinition, Sample
from plaid.storage.cgns.writer import (
    configure_dataset_card as configure_cgns_dataset_card,
)
from plaid.storage.cgns.writer import (
    generate_datasetdict_to_disk as generate_cgns_datasetdict_to_disk,
)
from plaid.storage.cgns.writer import (
    push_local_datasetdict_to_hub as push_local_cgns_datasetdict_to_hub,
)
from plaid.storage.common.preprocessor import preprocess
from plaid.storage.common.reader import (
    load_infos_from_disk,
    load_metadata_from_disk,
    load_problem_definitions_from_disk,
)
from plaid.storage.common.writer import (
    push_infos_to_hub,
    push_metadata_to_hub,
    push_problem_definitions_to_hub,
    save_infos_to_disk,
    save_metadata_to_disk,
    save_problem_definitions_to_disk,
)
from plaid.storage.hf_datasets.writer import (
    configure_dataset_card as configure_hf_dataset_card,
)
from plaid.storage.hf_datasets.writer import (
    generate_datasetdict_to_disk as generate_hf_datasetdict_to_disk,
)
from plaid.storage.hf_datasets.writer import (
    push_local_datasetdict_to_hub as push_local_hf_datasetdict_to_hub,
)
from plaid.storage.zarr.writer import (
    configure_dataset_card as configure_zarr_dataset_card,
)
from plaid.storage.zarr.writer import (
    generate_datasetdict_to_disk as generate_zarr_datasetdict_to_disk,
)
from plaid.storage.zarr.writer import (
    push_local_datasetdict_to_hub as push_local_zarr_datasetdict_to_hub,
)

logger = logging.getLogger(__name__)

AVAILABLE_BACKENDS = ["cgns", "hf_datasets", "zarr"]

generate_datasetdict_to_disk = {
    "hf_datasets": generate_hf_datasetdict_to_disk,
    "zarr": generate_zarr_datasetdict_to_disk,
    "cgns": generate_cgns_datasetdict_to_disk,
}

push_local_datasetdict_to_hub = {
    "hf_datasets": push_local_hf_datasetdict_to_hub,
    "zarr": push_local_zarr_datasetdict_to_hub,
    "cgns": push_local_cgns_datasetdict_to_hub,
}

configure_dataset_card = {
    "hf_datasets": configure_hf_dataset_card,
    "zarr": configure_zarr_dataset_card,
    "cgns": configure_cgns_dataset_card,
}


def _check_folder(output_folder: Path, overwrite: bool) -> None:
    """Check and prepare the output folder for dataset saving.

    This function ensures the output directory is ready for writing. If the directory exists
    and overwrite is True, it removes the existing directory. If it exists and is not empty
    without overwrite, it raises an error.

    Args:
        output_folder: Path to the output directory to check/prepare.
        overwrite: If True, removes existing directory if it exists.
    """
    if output_folder.is_dir():
        if overwrite:
            shutil.rmtree(output_folder)
            logger.warning(f"Existing {output_folder} directory has been reset.")
        elif any(output_folder.iterdir()):
            raise ValueError(
                f"directory {output_folder} already exists and is not empty. Set `overwrite` to True if needed."
            )


def save_to_disk(
    output_folder: Union[str, Path],
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    backend: str,
    infos: Optional[dict[str, Any]] = None,
    pb_defs: Optional[Union[ProblemDefinition, Iterable[ProblemDefinition]]] = None,
    gen_kwargs: Optional[dict[str, dict[str, Any]]] = None,
    num_proc: int = 1,
    verbose: bool = False,
    overwrite: bool = False,
) -> None:
    """Save a PLAID dataset to local disk using the specified backend.

    This function preprocesses the dataset generators, extracts schemas, and saves the dataset
    to disk using the chosen backend. It also saves metadata, infos, and problem definitions.

    Args:
        output_folder: Path to the output directory where the dataset will be saved.
        generators: Dictionary mapping split names to sample generators.
        backend: Storage backend to use ('cgns', 'hf_datasets', or 'zarr').
        infos: Optional additional information to save with the dataset.
        pb_defs: Optional problem definitions to save.
        gen_kwargs: Optional keyword arguments for the generators.
        num_proc: Number of processes to use for preprocessing.
        verbose: If True, enables verbose output during processing.
        overwrite: If True, overwrites existing output directory.
    """
    assert backend in AVAILABLE_BACKENDS, (
        f"backend {backend} not among available ones: {AVAILABLE_BACKENDS}"
    )

    output_folder = Path(output_folder)

    _check_folder(output_folder, overwrite)

    flat_cst, variable_schema, constant_schema, split_n_samples, cgns_types = (
        preprocess(
            generators, gen_kwargs=gen_kwargs, num_proc=num_proc, verbose=verbose
        )
    )

    generate_datasetdict_to_disk[backend](
        output_folder,
        generators,
        variable_schema,
        gen_kwargs=gen_kwargs,
        num_proc=num_proc,
        verbose=verbose,
    )

    save_metadata_to_disk(
        output_folder, flat_cst, variable_schema, constant_schema, cgns_types
    )

    infos = infos.copy() if infos else {}
    infos.setdefault("num_samples", split_n_samples)
    infos.setdefault("storage_backend", backend)
    infos.setdefault("plaid", {"version": str(Version(plaid.__version__))})

    save_infos_to_disk(output_folder, infos)

    if pb_defs is not None:
        save_problem_definitions_to_disk(output_folder, pb_defs)


def push_to_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    num_workers: int = 1,
    viewer: bool = False,
    pretty_name: Optional[str] = None,
    dataset_long_description: Optional[str] = None,
    illustration_urls: Optional[list[str]] = None,
    arxiv_paper_urls: Optional[list[str]] = None,
) -> None:  # pragma: no cover
    """Push a local PLAID dataset to Hugging Face Hub.

    This function uploads a previously saved dataset from local disk to Hugging Face Hub,
    including data, metadata, infos, and problem definitions. It automatically detects the
    backend used for saving and configures the dataset card.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name').
        local_dir: Local directory containing the saved dataset.
        num_workers: Number of workers for parallel upload.
        viewer: If True, enables dataset viewer on Hub.
        pretty_name: Optional pretty name for the dataset.
        dataset_long_description: Optional detailed description.
        illustration_urls: Optional list of illustration URLs.
        arxiv_paper_urls: Optional list of arXiv paper URLs.
    """
    pb_defs = load_problem_definitions_from_disk(local_dir)
    flat_cst, variable_schema, constant_schema, cgns_types = load_metadata_from_disk(
        local_dir
    )
    infos = load_infos_from_disk(local_dir)

    backend = infos["storage_backend"]

    push_local_datasetdict_to_hub[backend](repo_id, local_dir, num_workers=num_workers)
    configure_dataset_card[backend](
        repo_id,
        infos,
        local_dir,
        variable_schema,
        viewer,
        pretty_name,
        dataset_long_description,
        illustration_urls,
        arxiv_paper_urls,
    )

    push_metadata_to_hub(
        repo_id, flat_cst, variable_schema, constant_schema, cgns_types
    )
    push_infos_to_hub(repo_id, infos)
    if pb_defs is not None:
        push_problem_definitions_to_hub(repo_id, pb_defs)
