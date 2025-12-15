# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Common preprocessing utilities.

This module provides utilities for preprocessing PLAID samples into formats suitable
for storage, including flattening CGNS trees, inferring data types, and handling
parallel processing of sample shards.
"""

import hashlib
import multiprocessing as mp
import sys
import traceback
from queue import Empty
from typing import Any, Callable, Generator, Optional, Union

import numpy as np
from tqdm import tqdm

from plaid import Sample
from plaid.storage.common.tree_handling import flatten_cgns_tree
from plaid.types import IndexType


def infer_dtype(value: Any) -> dict[str, int | str]:
    """Infer canonical dtype schema from a value."""
    if value is None:  # pragma: no cover
        return {"dtype": "null", "ndim": 0}

    # Scalars
    if np.isscalar(value):  # pragma: no cover
        raise ValueError("CGNS should return arrays")

    # Arrays / lists
    elif isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value)
        dtype = arr.dtype
        if np.issubdtype(dtype, np.floating):
            dt = "float32"
        elif np.issubdtype(dtype, np.int32):
            dt = "int32"
        elif np.issubdtype(dtype, np.int64):
            dt = "int64"
        elif np.issubdtype(dtype, np.str_):
            dt = "string"
        else:  # pragma: no cover
            raise ValueError(f"Unrecognized scalar dtype: {dtype}")
        return {"dtype": dt, "ndim": arr.ndim}
        # arr = np.array(value)
        # return {"dtype": str(arr.dtype), "ndim": arr.ndim}

    raise TypeError(f"Unsupported type: {type(value)}")  # pragma: no cover


def build_sample_dict(
    sample: Sample,
) -> tuple[dict[str, Any], set[str], dict[str, str]]:
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
            - sample_dict (dict[str, Any]): Mapping of flattened CGNS paths to either a
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

    sample_dict = {}

    for path in all_paths:
        sample_dict[path] = None
        sample_dict[path + "_times"] = None

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
                sample_dict[path] = np.hstack(values_acc)
            except Exception:  # pragma: no cover
                sample_dict[path] = np.concatenate(
                    [np.atleast_1d(x) for x in values_acc]
                )

            if len(known_values) == 1:
                for t in times_acc:
                    t[-1] = -1
            sample_dict[path + "_times"] = np.array(times_acc).flatten()
        else:
            sample_dict[path] = None
            sample_dict[path + "_times"] = None

    # Convert lists to numpy arrays
    for k, v in sample_dict.items():
        if isinstance(v, list):
            sample_dict[k] = np.array(v)  # pragma: no cover

    return sample_dict, all_paths, sample_cgns_types


def _hash_value(value: Any) -> str:
    """Compute a hash for a value for deduplication.

    Args:
        value: The value to hash (np.ndarray or basic types).

    Returns:
        str: The MD5 hash of the value.
    """
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
    dict[str, Any],
    dict[str, dict[str, Union[str, bool, int]]],
    int,
]:
    """Process a single shard of sample ids and collect per-shard metadata.

    This function drives a shard-level pass over samples produced by `generator_fn`.
    For each sample it:
    - flattens the sample into Hugging Face friendly arrays (build_sample_dict),
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
            - shard_global_feature_types (dict[str, Union[Value, Sequence]]): Inferred feature types per path.
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
        generator = generator_fn([shard_ids])  # pragma: no cover
    else:
        generator = generator_fn()

    n_samples = 0
    for sample in generator:
        sample_dict, all_paths, sample_cgns_types = build_sample_dict(sample)

        split_all_paths.update(sample_dict.keys())
        shard_global_cgns_types.update(sample_cgns_types)

        # Feature type inference
        for path in all_paths:
            value = sample_dict[path]
            if value is None:
                continue
            inferred_dtype = infer_dtype(value)
            if path not in shard_global_feature_types:
                shard_global_feature_types[path] = inferred_dtype
            elif shard_global_feature_types[path] != inferred_dtype:
                raise ValueError(
                    f"Feature type mismatch for {path} in shard"
                )  # pragma: no cover

        # Constant detection using **hash only**
        for path, value in sample_dict.items():
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
    generator_fn: Callable[..., Any],
    progress_queue: Any,
    n_proc: int,
    shard_ids: Optional[list[IndexType]],
) -> Any:  # pragma: no cover
    """Debug wrapper for process_shard that prints exceptions.

    Args:
        generator_fn: The generator function.
        progress_queue: Queue for progress tracking.
        n_proc: Number of processes.
        shard_ids: List of shard IDs.

    Returns:
        The result of process_shard.
    """
    try:
        return process_shard(generator_fn, progress_queue, n_proc, shard_ids)
    except Exception as e:
        print(f"Exception in worker for shards {shard_ids}: {e}", file=sys.stderr)
        traceback.print_exc()
        raise  # re-raise to propagate to main process


def preprocess_splits(
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    gen_kwargs: Optional[dict[str, dict[str, Any]]] = None,
    num_proc: int = 1,
    verbose: bool = True,
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, Any]],
    dict[str, set[str]],
    dict[str, str],
    dict[str, Any],
    dict[str, int],
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
        num_proc (int, optional):
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
            - split_n_samples (dict[str, int]):
                For each split, the total number of samples processed.

    Raises:
        ValueError: If inconsistent feature types or CGNS types are detected across shards/splits.
    """
    global_cgns_types = {}
    global_feature_types = {}
    split_flat_cst = {}
    split_var_path = {}
    split_all_paths = {}
    split_n_samples = {}

    gen_kwargs_ = gen_kwargs or {split_name: {} for split_name in generators.keys()}

    for split_name, generator_fn in generators.items():
        shards_ids_list = gen_kwargs_[split_name].get("shards_ids", [None])
        n_proc = max(1, num_proc or len(shards_ids_list))

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

            for path, inferred_dtype in shard_features.items():
                if path not in global_feature_types:
                    global_feature_types[path] = inferred_dtype
                elif global_feature_types[path] != inferred_dtype:
                    raise ValueError(  # pragma: no cover
                        f"Feature type mismatch for {path} in split {split_name}"
                    )

            for path, entry in shard_constants.items():
                if path not in split_constant_hashes:
                    split_constant_hashes[path] = entry
                else:  # pragma: no cover
                    existing = split_constant_hashes[path]
                    existing["hashes"].update(entry["hashes"])
                    existing["count"] += entry["count"]

            n_samples_total += n_samples

        split_n_samples[split_name] = n_samples_total

        # Determine truly constant paths (same hash across all samples)
        constant_paths = [
            p
            for p, entry in split_constant_hashes.items()
            if len(entry["hashes"]) == 1 and entry["count"] == n_samples_total
        ]

        # Retrieve **values** only for constant paths from first sample
        if gen_kwargs:
            first_sample = next(generator_fn([shards_ids_list[0]]))  # pragma: no cover
        else:
            first_sample = next(generator_fn())
        sample_dict, _, _ = build_sample_dict(first_sample)

        split_flat_cst[split_name] = {p: sample_dict[p] for p in sorted(constant_paths)}
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
        split_n_samples,
    )


def preprocess(
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    gen_kwargs: Optional[dict[str, dict[str, Any]]] = None,
    num_proc: int = 1,
    verbose: bool = True,
):
    """Preprocess generators to extract schemas and metadata.

    Args:
        generators: Dict of split generators.
        gen_kwargs: Optional generator kwargs for parallel processing.
        num_proc: Number of processes.
        verbose: Whether to show progress.

    Returns:
        tuple: (split_flat_cst, variable_schema, constant_schema, split_n_samples, global_cgns_types)
    """
    assert (gen_kwargs is None and num_proc == 1) or (
        gen_kwargs is not None and num_proc > 1
    ), (
        "Invalid configuration: either provide only `generators` with "
        "`num_proc == 1`, or provide `gen_kwargs` with "
        "`num_proc > 1`."
    )

    (
        split_all_paths,
        split_flat_cst,
        split_var_path,
        global_cgns_types,
        global_feature_types,
        split_n_samples,
    ) = preprocess_splits(generators, gen_kwargs, num_proc, verbose)

    # --- build features ---
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
    first_split, first_value = next(iter(cst_features.items()))
    for split, value in cst_features.items():
        assert value == first_value, (
            f"cst_features differ for split '{split}' (vs '{first_split}')"
        )
    cst_features = first_value

    # var_features = [path for path in var_features if not path.endswith("_times")]
    # cst_features = [path for path in cst_features if not path.endswith("_times")]

    def _build_schema(feature_list: list[str]) -> dict:
        schema = {}
        for path in feature_list:
            if path.endswith("_times"):
                schema[path] = {
                    "dtype": "float64",
                    "ndim": 1,
                }  # pragma: no cover
            elif path in global_feature_types:
                schema[path] = global_feature_types[path]
            else:
                schema[path] = {"dtype": None}
        return schema

    variable_schema = _build_schema(var_features)
    constant_schema = _build_schema(cst_features)

    return (
        split_flat_cst,
        variable_schema,
        constant_schema,
        split_n_samples,
        global_cgns_types,
    )
