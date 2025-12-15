"""Zarr dataset writer module.

This module provides functionality for writing and managing datasets in Zarr format
for the PLAID library. It includes utilities for generating datasets from sample
generators, saving them to disk with optimized chunking, uploading to Hugging Face
Hub, and configuring dataset cards with metadata and usage examples.

Key features:
- Parallel and sequential dataset generation from generators
- Automatic chunking for efficient storage
- Integration with Hugging Face Hub for dataset sharing
- Dataset card generation with splits, features, and documentation
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

import numpy as np
import yaml
import zarr
from huggingface_hub import DatasetCard, HfApi
from tqdm import tqdm

from plaid import Sample
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.types import IndexType


def _flatten_path(key: str) -> str:
    """Flattens a path key by replacing slashes with underscores.

    Args:
        key (str): The path key to flatten.

    Returns:
        str: The flattened key with slashes replaced by underscores.
    """
    return key.replace("/", "__")


def _auto_chunks(shape: tuple[int, ...], target_n: int) -> tuple[int, ...]:
    """Computes automatic chunk sizes for Zarr arrays based on shape and target size.

    Args:
        shape (tuple[int, ...]): The shape of the array.
        target_n (int): The target number of elements per chunk.

    Returns:
        tuple[int, ...]: The computed chunk sizes.
    """
    # ensure pure Python ints
    target_n = int(target_n)
    shape = tuple(int(s) for s in shape)

    # elements in one "row"
    elems_per_slice = int(np.prod(shape[1:]) or 1)

    rows = max(1, target_n // elems_per_slice)
    rows = min(rows, shape[0])  # cannot exceed the dimension size

    return (rows,) + shape[1:]


def generate_datasetdict_to_disk(
    output_folder: Union[str, Path],
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    variable_schema: dict[str, dict],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    num_proc: int = 1,
    verbose: bool = False,
) -> None:
    """Generates and saves a dataset dictionary to disk in Zarr format.

    This function processes sample generators for different dataset splits,
    converts samples to dictionaries, and writes them to Zarr arrays on disk.
    It supports both sequential and parallel processing modes. In parallel mode,
    gen_kwargs must be provided with batch information for each split.

    Args:
        output_folder (Union[str, Path]): Base directory where the dataset will be saved.
            A 'data' subdirectory will be created inside this folder.
        generators (dict[str, Callable[..., Generator[Sample, None, None]]]):
            Dictionary mapping split names (e.g., "train", "test") to generator
            functions that yield Sample objects.
        variable_schema (dict[str, dict]): Schema describing the structure and types
            of variables/features in the samples.
        gen_kwargs (Optional[dict[str, dict[str, list[IndexType]]]]): Optional
            generator arguments for parallel processing. Must include "shards_ids"
            for each split when num_proc > 1. Required for parallel execution.
        num_proc (int, optional): Number of processes to use for parallel processing.
            Defaults to 1 (sequential). Must be > 1 only when gen_kwargs is provided.
        verbose (bool, optional): Whether to display progress bars during processing.
            Defaults to False.

    Returns:
        None: This function does not return a value; it writes the dataset directly
            to disk.
    """
    assert (gen_kwargs is None and num_proc == 1) or (
        gen_kwargs is not None and num_proc > 1
    ), (
        "Invalid configuration: either provide only `generators` with "
        "`num_proc == 1`, or provide `gen_kwargs` with "
        "`num_proc > 1`."
    )

    output_folder = Path(output_folder) / "data"
    output_folder.mkdir(exist_ok=True, parents=True)

    var_features_keys = list(variable_schema.keys())

    def worker_batch(
        split_root_path: str,
        gen_func: Callable[..., Generator[Sample, None, None]],
        var_features_keys: list[str],
        batch: list[IndexType],
        start_index: int,
        queue: mp.Queue,
    ) -> None:  # pragma: no cover
        """Processes a single batch and writes samples to Zarr.

        Args:
            split_root_path (str): Path to the Zarr group for the split.
            gen_func (Callable): Generator function for samples.
            var_features_keys (list[str]): List of feature keys.
            batch (list[IndexType]): Batch of sample IDs.
            start_index (int): Starting sample index.
            queue (mp.Queue): Queue for progress tracking.
        """
        split_root = zarr.open_group(split_root_path, mode="a")
        sample_counter = start_index

        for sample in gen_func([batch]):
            sample_dict, _, _ = build_sample_dict(sample)
            sample_data = {
                path: sample_dict.get(path, None) for path in var_features_keys
            }

            g = split_root.create_group(f"sample_{sample_counter:09d}")
            for key, value in sample_data.items():
                g.create_array(
                    _flatten_path(key),
                    data=value,
                    chunks=_auto_chunks(value.shape, 5_000_000),
                )  # chunks=value.shape

            sample_counter += 1
            queue.put(1)

    def tqdm_updater(
        total: int, queue: mp.Queue, desc: str = "Processing"
    ) -> None:  # pragma: no cover
        """Tqdm process that listens to the queue to update progress.

        Args:
            total (int): Total number of items to process.
            queue (mp.Queue): Queue to receive progress updates.
            desc (str): Description for the progress bar.
        """
        with tqdm(total=total, desc=desc, disable=not verbose) as pbar:
            finished = 0
            while finished < total:
                finished += queue.get()
                pbar.update(1)

    for split_name, gen_func in generators.items():
        split_root_path = str(output_folder / split_name)
        split_root = zarr.open_group(split_root_path, mode="w")

        gen_kwargs_ = gen_kwargs or {sn: {} for sn in generators.keys()}
        batch_ids_list = gen_kwargs_.get(split_name, {}).get("shards_ids", [])

        total_samples = sum(len(batch) for batch in batch_ids_list)

        if num_proc > 1 and batch_ids_list:  # pragma: no cover
            # Parallel execution
            queue = mp.Queue()
            tqdm_proc = mp.Process(
                target=tqdm_updater,
                args=(total_samples, queue, f"Writing {split_name} split"),
            )
            tqdm_proc.start()

            processes = []
            start_index = 0
            for batch in batch_ids_list:
                p = mp.Process(
                    target=worker_batch,
                    args=(
                        split_root_path,
                        gen_func,
                        var_features_keys,
                        batch,
                        start_index,
                        queue,
                    ),
                )
                p.start()
                processes.append(p)
                start_index += len(batch)

            for p in processes:
                p.join()

            tqdm_proc.join()

        else:
            # Sequential execution
            sample_counter = 0
            with tqdm(
                total=total_samples,
                desc=f"Writing {split_name} split",
                disable=not verbose,
            ) as pbar:
                for sample in gen_func():
                    sample_dict, _, _ = build_sample_dict(sample)
                    sample_data = {
                        path: sample_dict.get(path, None) for path in var_features_keys
                    }

                    g = split_root.create_group(f"sample_{sample_counter:09d}")
                    for key, value in sample_data.items():
                        if value is not None:
                            g.create_array(
                                _flatten_path(key),
                                data=value,
                                chunks=_auto_chunks(value.shape, 5_000_000),
                            )  # chunks=value.shape

                    sample_counter += 1
                    pbar.update(1)


def push_local_datasetdict_to_hub(
    repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
) -> None:  # pragma: no cover
    """Pushes a local dataset directory to Hugging Face Hub.

    This function uploads the contents of a local directory to a specified
    Hugging Face repository as a dataset. It uses the HfApi to handle large
    folder uploads with configurable parallelism.

    Args:
        repo_id (str): The Hugging Face repository ID where the dataset will be uploaded
            (e.g., "username/dataset_name").
        local_dir (str or Path): Path to the local directory containing the dataset files
            to upload.
        num_workers (int, optional): Number of worker threads to use for uploading.
            Defaults to 1.

    Returns:
        None: This function does not return a value; it uploads the dataset directly
            to Hugging Face Hub.
    """
    api = HfApi()
    api.upload_large_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=num_workers,
        ignore_patterns=["*.tmp"],
        allow_patterns=["data/*"],
    )


def configure_dataset_card(
    repo_id: str,
    infos: dict[str, dict[str, str]],
    local_dir: Union[str, Path],
    variable_schema: Optional[dict] = None,
    viewer: Optional[bool] = None,  # noqa: ARG001
    pretty_name: Optional[str] = None,
    dataset_long_description: Optional[str] = None,
    illustration_urls: Optional[list[str]] = None,
    arxiv_paper_urls: Optional[list[str]] = None,
) -> None:  # pragma: no cover
    """Configures and pushes a dataset card to Hugging Face Hub for a zarr backend dataset.

    This function generates a dataset card in YAML format with metadata, features,
    splits information, and usage examples. It automatically detects splits and
    sample counts from the local directory structure, then pushes the card to
    the specified Hugging Face repository.

    Args:
        repo_id (str): The Hugging Face repository ID where the dataset card will be pushed.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset metadata,
            including legal information like license.
        local_dir (Union[str, Path]): Path to the local directory containing the
            dataset files, expected to have a 'data' subdirectory with split folders.
        variable_schema (Optional[dict]): Schema describing the variables/features
            in the dataset, used to generate the features section in the card.
        viewer (Optional[bool]): Unused parameter for viewer configuration.
        pretty_name (Optional[str]): A human-readable name for the dataset to
            display in the card.
        dataset_long_description (Optional[str]): A detailed description of the
            dataset to include in the card.
        illustration_urls (Optional[list[str]]): List of URLs to images that
            illustrate the dataset, displayed in the card.
        arxiv_paper_urls (Optional[list[str]]): List of arXiv URLs for papers
            related to the dataset, included as sources.

    Returns:
        None: This function does not return a value; it pushes the dataset card
            directly to Hugging Face Hub.
    """

    def _dict_to_list_format(d: dict[str, Any]) -> str:
        """Converts a variable schema dict to YAML list format.

        Args:
            d (dict[str, Any]): Dictionary with 'dtype' and 'ndim' keys.

        Returns:
            str: YAML-formatted string representing the list structure.
        """
        dtype = d.get("dtype", "unknown")
        ndim = d.get("ndim", 1)

        lines = []
        current_indent = "    "

        for _ in range(ndim - 1):
            lines.append(f"{current_indent}list:")
            current_indent += "  "

        # last level contains the dtype as value
        lines.append(f"{current_indent}list: {dtype}")

        return "\n".join(lines)

    dataset_card_str = """---
task_categories:
- graph-ml
tags:
- physics learning
- geometry learning
---
"""
    local_folder = Path(local_dir)
    split_names = [p.name for p in (local_folder / "data").iterdir() if p.is_dir()]

    nbe_samples = {}
    num_bytes = {}
    size_bytes = 0
    for sn in split_names:
        nbe_samples[sn] = sum(
            1
            for p in (local_folder / "data" / f"{sn}").iterdir()
            if p.is_dir() and p.name.startswith("sample_")
        )
        num_bytes[sn] = sum(
            f.stat().st_size
            for f in (local_folder / "data" / f"{sn}").rglob("*")
            if f.is_file()
        )
        size_bytes += num_bytes[sn]

    lines = dataset_card_str.splitlines()
    lines = [s for s in lines if not s.startswith("license")]

    indices = [i for i, line in enumerate(lines) if line.strip() == "---"]

    assert len(indices) >= 2, (
        "Cannot find two instances of '---', you should try to update a correct dataset_card."
    )
    lines = lines[: indices[1] + 1]

    count = 6
    lines.insert(count, f"license: {infos['legal']['license']}")
    count += 1
    lines.insert(count, "viewer: false")
    count += 1
    if pretty_name:
        lines.insert(count, f"pretty_name: {pretty_name}")
        count += 1

    lines.insert(count, "dataset_info:")
    count += 1
    if variable_schema is not None:
        lines.insert(count, "  features:")
        count += 1
        for fn, type_ in variable_schema.items():
            lines.insert(count, f"  - name: {_flatten_path(fn)}")
            count += 1
            lines.insert(count, _dict_to_list_format(type_))
            count += 1
    lines.insert(count, "  splits:")
    count += 1
    for sn in split_names:
        lines.insert(count, f"    - name: {sn}")
        count += 1
        lines.insert(count, f"      num_bytes: {num_bytes[sn]}")
        count += 1
        lines.insert(count, f"      num_examples: {nbe_samples[sn]}")
        count += 1
    lines.insert(count, f"  download_size: {size_bytes}")
    count += 1
    lines.insert(count, f"  dataset_size: {size_bytes}")
    count += 1
    lines.insert(count, "configs:")
    count += 1
    lines.insert(count, "- config_name: default")
    count += 1
    lines.insert(count, "  data_files:")
    count += 1
    for sn in split_names:
        lines.insert(count, f"  - split: {sn}")
        count += 1
        lines.insert(count, f"    path: data/{sn}/*")
        count += 1

    str__ = "\n".join(lines) + "\n"

    if illustration_urls:
        str__ += "<p align='center'>\n"
        for url in illustration_urls:
            str__ += f"<img src='{url}' alt='{url}' width='1000'/>\n"
        str__ += "</p>\n\n"

    str__ += f"```yaml\n{yaml.dump(infos, sort_keys=False, allow_unicode=True)}\n```"

    str__ += """
This dataset was generated with [`plaid`](https://plaid-lib.readthedocs.io/), we refer to this documentation for additional details on how to extract data from `plaid_sample` objects.

The simplest way to use this dataset is to first download it:
```python
from plaid.storage import download_from_hub

repo_id = "channel/dataset"
local_folder = "downloaded_dataset"

download_from_hub(repo_id, local_folder)
```

Then, to iterate over the dataset and instantiate samples:
```python
from plaid.storage import init_from_disk

local_folder = "downloaded_dataset"
split_name = "train"

datasetdict, converterdict = init_from_disk(local_folder)

dataset = datasetdict[split]
converter = converterdict[split]

for i in range(len(dataset)):
    plaid_sample = converter.to_plaid(dataset, i)
```

It is possible to stream the data directly:
```python
from plaid.storage import init_streaming_from_hub

repo_id = "channel/dataset"

datasetdict, converterdict = init_streaming_from_hub(repo_id)

dataset = datasetdict[split]
converter = converterdict[split]

for sample_raw in dataset:
    plaid_sample = converter.sample_to_plaid(sample_raw)
```

Plaid samples' features can be retrieved like the following:
```python
from plaid.storage import load_problem_definitions_from_disk
local_folder = "downloaded_dataset"
pb_defs = load_problem_definitions_from_disk(local_folder)

# or
from plaid.storage import load_problem_definitions_from_hub
repo_id = "channel/dataset"
pb_defs = load_problem_definitions_from_hub(repo_id)


pb_def = pb_defs[0]

plaid_sample = ... # use a method from above to instantiate a plaid sample

for t in plaid_sample.get_all_time_values():
    for path in pb_def.get_in_features_identifiers():
        plaid_sample.get_feature_by_path(path=path, time=t)
    for path in pb_def.get_out_features_identifiers():
        plaid_sample.get_feature_by_path(path=path, time=t)
```
"""

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

    dataset_card = DatasetCard(str__)
    dataset_card.push_to_hub(repo_id)
