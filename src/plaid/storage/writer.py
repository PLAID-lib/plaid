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
from pathlib import Path
from typing import Any, Callable, Generator, Mapping, Optional, Sequence, Union

from packaging.version import Version


from ..containers.sample import Sample
from ..problem_definition import ProblemDefinition

from ..containers.utils import validate_required_infos
from .common.preprocessor import preprocess
from .common.reader import (
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

logger = logging.getLogger(__name__)


def _split_list(lst: list, n_splits: int) -> list[list]:
    """Split a list into n nearly-equal chunks."""
    if n_splits <= 1:
        return [lst]
    n = len(lst)
    k, m = divmod(n, n_splits)
    return [
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_splits)
    ]


class _SampleFuncGenerator:
    """Picklable generator callable that wraps a user-provided sample function.

    This class turns a simple ``sample_constructor(id) -> Sample`` into a generator
    compatible with the backend ``generate_to_disk`` interface.  It is defined
    at module level so that ``multiprocessing`` can pickle it.
    """

    def __init__(self, sample_constructor: Callable[[Any], Sample]) -> None:
        self._func = sample_constructor

    def __call__(
        self,
        shards_ids: Optional[list[list]] = None,
    ) -> Generator[Sample, None, None]:
        if shards_ids is None:
            shards_ids = [[]]
        for shard in shards_ids:
            for id_ in shard:
                yield self._func(id_)


def _build_gen_kwargs(
    ids: Mapping[str, Sequence],
    num_proc: int,
) -> dict[str, dict[str, Any]]:
    """Build ``gen_kwargs`` by sharding the ids for each split.

    Returns:
    -------
    gen_kwargs : dict
        Per-split kwargs with ``shards_ids`` containing the shard lists.
    """
    gen_kwargs: dict[str, dict[str, Any]] = {}

    for split_name, split_ids in ids.items():
        n_shards = min(max(1, num_proc), max(1, len(split_ids)))
        shards = [
            chunk for chunk in _split_list(list(split_ids), n_shards) if len(chunk) > 0
        ]
        gen_kwargs[split_name] = {"shards_ids": shards}

    return gen_kwargs


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
    sample_constructor: Callable[[Any], Sample],
    ids: Mapping[str, Sequence],
    backend: str = "hf_datasets",
    infos: Optional[dict[str, Any]] = None,
    pb_defs: Optional[Union[dict[str, ProblemDefinition], ProblemDefinition]] = None,
    num_proc: int = 1,
    verbose: bool = False,
    overwrite: bool = False,
) -> None:
    """Save a PLAID dataset to local disk using the specified backend.

    This function preprocesses the dataset, extracts schemas, and saves
    the dataset to disk using the chosen backend.  It also saves metadata, infos,
    and problem definitions.

    The user provides a simple function ``sample_constructor`` that takes a single
    identifier and returns a :class:`~plaid.Sample`, together with a dictionary
    ``ids`` mapping split names to sliceable sequences of identifiers.
    PLAID handles iteration, generator creation, and parallel sharding
    internally.

    Example::

        from plaid import Sample
        from plaid.storage import save_to_disk

        def sample_constructor(file_path):
            sample = Sample()
            sample.add_tree(load_my_data(file_path))
            return sample

        save_to_disk(
            "output/",
            sample_constructor=sample_constructor,
            ids={
                "train": train_file_paths,
                "test":  test_file_paths,
            },
            num_proc=6,
        )

    Args:
        output_folder: Path to the output directory where the dataset will be saved.
        sample_constructor: A callable that takes a single identifier (of any type)
            and returns a :class:`~plaid.Sample`.
        ids: Dictionary mapping split names (e.g. ``"train"``, ``"test"``) to
            sliceable sequences of sample identifiers.  Each sequence must
            support ``__getitem__`` and ``__len__`` (list, tuple, numpy array,
            …).  The identifiers can be of any type: integers, file paths,
            strings, tuples, etc.
        backend: Storage backend to use (``'cgns'``, ``'hf_datasets'``, or
            ``'zarr'``).
        infos: Optional additional information to save with the dataset.
        pb_defs: Optional problem definitions to save.
        num_proc: Number of processes to use for parallel writing.  When
            ``num_proc > 1`` PLAID automatically shards the identifier
            sequences and distributes work across workers.
        verbose: If True, enables verbose output during processing.
        overwrite: If True, overwrites existing output directory.
    """
    assert backend in available_backends(), (
        f"backend {backend} not among available ones: {available_backends()}"
    )
    if infos:
        validate_required_infos(infos)

    # ---- validate ids: must be sliceable sequences ---------------------------
    for split_name, split_ids in ids.items():
        if not (hasattr(split_ids, "__getitem__") and hasattr(split_ids, "__len__")):
            raise TypeError(
                f"ids for split '{split_name}' must be a sliceable sequence "
                f"(with __getitem__ and __len__), got {type(split_ids).__name__}. "
                f"Use a list, tuple, or numpy array of sample identifiers."
            )

    # ---- build generators from sample_constructor -----------------------------------
    generators: dict[str, Callable[..., Generator[Sample, None, None]]] = {}
    for split_name in ids:
        generators[split_name] = _SampleFuncGenerator(sample_constructor)

    # ---- auto-shard when running in parallel ---------------------------------
    gen_kwargs: Optional[dict[str, dict[str, Any]]] = None
    if num_proc > 1:
        gen_kwargs = _build_gen_kwargs(ids, num_proc)
    else:
        # For sequential execution, wrap ids into a single shard so the
        # generator receives them via shards_ids
        gen_kwargs = {
            split_name: {"shards_ids": [list(split_ids)]}
            for split_name, split_ids in ids.items()
        }

    output_folder = Path(output_folder)
    _check_folder(output_folder, overwrite)

    flat_cst, variable_schema, constant_schema, num_samples, cgns_types = preprocess(
        generators, gen_kwargs=gen_kwargs, num_proc=num_proc, verbose=verbose
    )

    save_metadata_to_disk(
        output_folder, flat_cst, variable_schema, constant_schema, cgns_types
    )

    infos = infos.copy() if infos else {}
    infos.setdefault("num_samples", num_samples)
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


# ---------------------------------------------------------------------------
# Hub push
# ---------------------------------------------------------------------------


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
