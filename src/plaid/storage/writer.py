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

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import logging
import shutil
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union, cast

from packaging.version import Version

import plaid
from plaid import ProblemDefinition, Sample
from plaid.containers.utils import validate_required_infos
from plaid.storage.common.preprocessor import preprocess
from plaid.storage.common.reader import (
    load_infos_from_disk,
)
from plaid.storage.common.writer import (
    push_infos_to_hub,
    push_local_metadata_to_hub,
    push_local_problem_definitions_to_hub,
    save_infos_to_disk,
    save_metadata_to_disk,
    save_problem_definitions_to_disk,
)
from plaid.storage.registry import available_backends, get_backend
from plaid.types import IndexType

logger = logging.getLogger(__name__)


def _split_list(lst: list[IndexType], n_splits: int) -> list[list[IndexType]]:
    """Split a list into n nearly-equal chunks."""
    if n_splits <= 1:
        return [lst]
    n = len(lst)
    k, m = divmod(n, n_splits)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_splits)
    ]


def _build_parallel_gen_kwargs(
    split_ids: dict[str, list[IndexType]],
    num_proc: int,
) -> dict[str, dict[str, list[list[IndexType]]]]:
    """Build internal per-split shard ids from split ids and process count."""
    gen_kwargs: dict[str, dict[str, list[list[IndexType]]]] = {}
    for split_name, ids in split_ids.items():
        n_shards = min(max(1, num_proc), max(1, len(ids)))
        shards_ids = [chunk for chunk in _split_list(ids, n_shards) if len(chunk) > 0]
        gen_kwargs[split_name] = {"shards_ids": shards_ids}
    return gen_kwargs


def _generator_from_split_ids(
    shards_ids: Optional[list[list[IndexType]]] = None,
    *,
    gen_func: Callable[..., Generator[Sample, None, None]],
    split_ids: list[IndexType],
) -> Generator[Sample, None, None]:
    """Adapter generator used by high-level index-based APIs."""
    if shards_ids is None:
        yield from gen_func(split_ids)
    else:
        for ids in shards_ids:
            yield from gen_func(ids)


def _wrap_generators_with_split_ids(
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    split_ids: dict[str, list[IndexType]],
) -> dict[str, Callable[..., Generator[Sample, None, None]]]:
    """Wrap generators so users can provide `ids` while internals keep shard semantics.

    Input generators are expected to accept one argument `ids` (a list of indices)
    and yield samples. With `split_n_samples`, these ids are local split indices in
    [0, ..., n-1].

    The wrapper supports both:
    - sequential internal calls with no args, by using full split ids,
    - parallel internal calls with `shards_ids`, by iterating over shard chunks.
    """
    wrapped: dict[str, Callable[..., Generator[Sample, None, None]]] = {}

    for split_name, gen_func in generators.items():
        split_ids_ = split_ids[split_name]
        wrapped[split_name] = partial(
            _generator_from_split_ids,
            gen_func=gen_func,
            split_ids=split_ids_,
        )

    return wrapped


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
    backend: str = "hf_datasets",
    infos: Optional[dict[str, Any]] = None,
    pb_defs: Optional[Union[dict[str, ProblemDefinition], ProblemDefinition]] = None,
    split_n_samples: Optional[dict[str, int]] = None,
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
            Each generator must accept one positional argument `ids` and yield
            corresponding samples.
        backend: Storage backend to use ('cgns', 'hf_datasets', or 'zarr').
        infos: Optional additional information to save with the dataset.
        pb_defs: Optional problem definitions to save.
        split_n_samples: Optional mapping split -> number of samples. When provided,
            PLAID builds internal shard ids using local indices [0..n-1] and handles
            sharding internally for parallel writing. In this mode, generator `ids`
            are local indices in each split.
        gen_kwargs: Optional keyword arguments for the generators.
        num_proc: Number of processes to use for preprocessing.
        verbose: If True, enables verbose output during processing.
        overwrite: If True, overwrites existing output directory.
    """
    assert backend in available_backends(), (
        f"backend {backend} not among available ones: {available_backends()}"
    )
    if infos:
        validate_required_infos(infos)

    if split_n_samples is not None and gen_kwargs is not None:
        raise ValueError(
            "Provide either `split_n_samples` (high-level API) or `gen_kwargs` (advanced API), not both."
        )

    if split_n_samples is not None:
        missing = set(generators.keys()) - set(split_n_samples.keys())
        if missing:
            raise ValueError(
                f"Missing split sizes for splits: {sorted(missing)}. Expected one size per generator split."
            )
        extra = set(split_n_samples.keys()) - set(generators.keys())
        if extra:
            raise ValueError(
                f"Unexpected split size keys not found in generators: {sorted(extra)}."
            )
        if any(n < 0 for n in split_n_samples.values()):
            raise ValueError("split_n_samples values must be >= 0.")

        split_ids = {
            split_name: [cast(IndexType, i) for i in range(split_n)]
            for split_name, split_n in split_n_samples.items()
        }
        generators = _wrap_generators_with_split_ids(generators, split_ids)
        if num_proc > 1:
            gen_kwargs = _build_parallel_gen_kwargs(split_ids, num_proc)

    output_folder = Path(output_folder)
    _check_folder(output_folder, overwrite)

    flat_cst, variable_schema, constant_schema, split_n_samples, cgns_types = (
        preprocess(
            generators, gen_kwargs=gen_kwargs, num_proc=num_proc, verbose=verbose
        )
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

    backend_spec = get_backend(backend)
    backend_spec.generate_to_disk(
        output_folder,
        generators,
        variable_schema,
        gen_kwargs=gen_kwargs,
        num_proc=num_proc,
        verbose=verbose,
    )


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
    infos = load_infos_from_disk(local_dir)

    validate_required_infos(infos)

    backend = infos["storage_backend"]

    backend_spec = get_backend(backend)
    backend_spec.push_local_to_hub(repo_id, local_dir, num_workers=num_workers)
    backend_spec.configure_dataset_card(
        repo_id,
        infos,
        local_dir,
        # variable_schema,
        viewer,
        pretty_name,
        dataset_long_description,
        illustration_urls,
        arxiv_paper_urls,
    )

    push_local_metadata_to_hub(repo_id, local_dir)

    push_infos_to_hub(repo_id, infos)

    push_local_problem_definitions_to_hub(repo_id, local_dir)
