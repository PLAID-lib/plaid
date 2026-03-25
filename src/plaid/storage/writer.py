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
from typing import Any, Callable, Generator, Optional, Union

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


def _extract_partial_data(
    gen: Callable,
) -> Optional[list]:
    """Extract the first positional argument from a functools.partial.

    Returns the data (first arg) if *gen* is a ``functools.partial``, otherwise
    ``None``.
    """
    if isinstance(gen, partial) and gen.args:
        data = gen.args[0]
        # Ensure it's something we can slice
        if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
            return data
    return None


def _build_gen_kwargs_from_partials(
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    num_proc: int,
) -> tuple[
    dict[str, Callable[..., Generator[Sample, None, None]]],
    dict[str, dict[str, Any]],
]:
    """Build ``gen_kwargs`` by sharding the data closed over in each ``partial``.

    For each split the first positional argument of the ``partial`` is extracted,
    sliced into ``num_proc`` shards, and new ``partial`` generators are created
    that accept a single ``shard`` keyword argument.

    Returns:
    -------
    wrapped_generators : dict
        New generators (one per split) that accept ``shard=<list>`` and yield
        samples for that shard.
    gen_kwargs : dict
        Per-split kwargs suitable for the backend ``generate_to_disk`` calls,
        with ``shards_ids`` containing the shard lists.
    """
    wrapped_generators: dict[str, Callable[..., Generator[Sample, None, None]]] = {}
    gen_kwargs: dict[str, dict[str, Any]] = {}

    for split_name, gen in generators.items():
        data = _extract_partial_data(gen)
        if data is None:
            raise TypeError(
                f"Generator for split '{split_name}' must be a functools.partial "
                f"whose first positional argument is a sliceable sequence of sample "
                f"identifiers (e.g. partial(my_gen, my_ids)). "
                f"Got {type(gen).__name__} instead."
            )

        n_shards = min(max(1, num_proc), max(1, len(data)))
        shards = [
            chunk for chunk in _split_list(list(data), n_shards) if len(chunk) > 0
        ]

        # The underlying function (unwrapped from partial)
        base_func = gen.func
        extra_args = gen.args[1:]  # any additional positional args
        extra_kwargs = gen.keywords  # any additional keyword args

        def _shard_generator(
            *,
            shards_ids: list[list],
            _base_func: Callable = base_func,
            _extra_args: tuple = extra_args,
            _extra_kwargs: dict = extra_kwargs,
        ) -> Generator[Sample, None, None]:
            for shard in shards_ids:
                yield from _base_func(shard, *_extra_args, **_extra_kwargs)

        wrapped_generators[split_name] = _shard_generator
        gen_kwargs[split_name] = {"shards_ids": shards}

    return wrapped_generators, gen_kwargs


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
    num_proc: int = 1,
    verbose: bool = False,
    overwrite: bool = False,
    # --- advanced ---------------------------------------------------------------
    gen_kwargs: Optional[dict[str, dict[str, Any]]] = None,
) -> None:
    """Save a PLAID dataset to local disk using the specified backend.

    This function preprocesses the dataset generators, extracts schemas, and saves
    the dataset to disk using the chosen backend.  It also saves metadata, infos,
    and problem definitions.

    Generators must be ``functools.partial`` objects whose first positional
    argument is a **sliceable sequence** of sample identifiers or data
    references — anything with ``__getitem__`` and ``__len__``.  This can be
    a list of integers, file paths, strings, pre-loaded data objects, a numpy
    array, etc.  PLAID calls the underlying function with the full sequence
    for sequential execution and with sub-sequences (shards) for parallel
    execution — the user code is identical in both cases::

        from functools import partial

        def my_generator(ids):
            for i in ids:
                yield make_sample(i)

        generators = {
            "train": partial(my_generator, train_ids),
            "test":  partial(my_generator, test_ids),
        }

        # Sequential
        save_to_disk("out", generators)

        # Parallel — same generators, only num_proc changes
        save_to_disk("out", generators, num_proc=6)

    Args:
        output_folder: Path to the output directory where the dataset will be saved.
        generators: Dictionary mapping split names to ``functools.partial``
            generator callables.  Each underlying function must accept a
            sequence of sample identifiers as its first argument and yield
            the corresponding ``Sample`` objects.
        backend: Storage backend to use (``'cgns'``, ``'hf_datasets'``, or
            ``'zarr'``).
        infos: Optional additional information to save with the dataset.
        pb_defs: Optional problem definitions to save.
        num_proc: Number of processes to use for parallel writing.  When
            ``num_proc > 1`` PLAID automatically shards the identifier
            sequences and distributes work across workers.
        verbose: If True, enables verbose output during processing.
        overwrite: If True, overwrites existing output directory.
        gen_kwargs: *Advanced* — explicit per-split generator keyword
            arguments.  When provided PLAID skips automatic sharding and
            forwards these kwargs directly to the backend.
    """
    assert backend in available_backends(), (
        f"backend {backend} not among available ones: {available_backends()}"
    )
    if infos:
        validate_required_infos(infos)

    # ---- validate generators: must be partial with sliceable first arg -------
    for split_name, gen in generators.items():
        if not isinstance(gen, partial):
            raise TypeError(
                f"Generator for split '{split_name}' must be a functools.partial, "
                f"got {type(gen).__name__}. "
                f"Wrap your generator function: partial(my_generator, my_ids)."
            )
        if not gen.args:
            raise TypeError(
                f"Generator for split '{split_name}' is a functools.partial but has "
                f"no positional arguments. The first positional argument must be a "
                f"sliceable sequence of sample identifiers "
                f"(e.g. partial(my_gen, my_ids))."
            )
        data = gen.args[0]
        if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
            raise TypeError(
                f"Generator for split '{split_name}': the first positional argument "
                f"must be a sliceable sequence (with __getitem__ and __len__), "
                f"got {type(data).__name__}. "
                f"Use a list, tuple, or numpy array of sample identifiers."
            )

    # ---- auto-shard from partial when running in parallel --------------------
    if gen_kwargs is None and num_proc > 1:
        generators, gen_kwargs = _build_gen_kwargs_from_partials(generators, num_proc)

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
