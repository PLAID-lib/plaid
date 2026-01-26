"""WebDataset writer module.

This module provides functionality for writing and managing datasets in WebDataset format
for the PLAID library. It includes utilities for generating datasets from sample
generators, saving them to tar archives, uploading to Hugging Face Hub, and configuring
dataset cards with metadata and usage examples.

Key features:
- Parallel and sequential dataset generation from generators
- Tar-based storage format for streaming compatibility
- Integration with Hugging Face Hub for dataset sharing
- Dataset card generation with splits, features, and documentation
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import io
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

import numpy as np
import yaml
import webdataset as wds
from huggingface_hub import DatasetCard, HfApi
from tqdm import tqdm

from plaid import Sample
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.types import IndexType


def _write_sample_to_tar(
    tar_writer: wds.TarWriter,
    sample: Sample,
    var_features_keys: list[str],
    sample_idx: int,
) -> None:
    """Write a single PLAID sample to a WebDataset tar archive.

    This function serializes one Sample instance into a set of files in a tar archive.
    Each sample is written with a common basename (e.g., sample_000000000) and different
    extensions for different data types:
    - .npy files for numpy arrays (variable features)
    - .json for metadata and non-array data

    Args:
        tar_writer: WebDataset TarWriter instance for writing to tar.
        sample: PLAID Sample object to serialize.
        var_features_keys: List of feature paths to extract and write.
        sample_idx: Global index of the sample for naming.
    """
    sample_dict, _, _ = build_sample_dict(sample)
    sample_data = {path: sample_dict.get(path, None) for path in var_features_keys}

    # Create a dictionary to hold the sample
    basename = f"sample_{sample_idx:09d}"
    sample_files = {"__key__": basename}

    # Separate arrays and metadata
    # Track which base features have values
    features_with_values = set()

    for key, value in sample_data.items():
        # Skip _times keys for now, we'll handle them after
        if key.endswith("_times"):
            continue

        if value is None:
            continue

        # Mark that this feature has a value
        features_with_values.add(key)

        # Convert numpy arrays to bytes
        if isinstance(value, np.ndarray):
            # Save numpy array to bytes
            buffer = io.BytesIO()
            np.save(buffer, value)
            buffer.seek(0)
            # Use key as filename with .npy extension, replacing / with __
            safe_key = key.replace("/", "__")
            sample_files[f"{safe_key}.npy"] = buffer.read()
        else:
            # Store non-arrays as JSON metadata
            if not hasattr(_write_sample_to_tar, "_metadata_keys"):
                _write_sample_to_tar._metadata_keys = []
            if key not in _write_sample_to_tar._metadata_keys:
                _write_sample_to_tar._metadata_keys.append(key)

    # Now add _times only for features that have values
    for key, value in sample_data.items():
        if not key.endswith("_times"):
            continue

        base_feature = key[:-6]  # Remove "_times" suffix

        # Only write _times if the base feature has a value
        if base_feature in features_with_values and value is not None:
            buffer = io.BytesIO()
            np.save(buffer, value)
            buffer.seek(0)
            safe_key = key.replace("/", "__")
            sample_files[f"{safe_key}.npy"] = buffer.read()

    # Write all files for this sample to tar
    tar_writer.write(sample_files)


def generate_datasetdict_to_disk(
    output_folder: Union[str, Path],
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    variable_schema: dict[str, dict],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    num_proc: int = 1,
    verbose: bool = False,
) -> None:
    """Generates and saves a dataset dictionary to disk in WebDataset format.

    This function processes sample generators for different dataset splits,
    converts samples to dictionaries, and writes them to tar archives on disk.
    It supports both sequential and parallel processing modes.

    Args:
        output_folder: Base directory where the dataset will be saved.
            A 'data' subdirectory will be created inside this folder.
        generators: Dictionary mapping split names to generator functions
            that yield Sample objects.
        variable_schema: Schema describing the structure and types of
            variables/features in the samples.
        gen_kwargs: Optional generator arguments for parallel processing.
            Must include "shards_ids" for each split when num_proc > 1.
        num_proc: Number of processes to use for parallel processing.
            Defaults to 1 (sequential).
        verbose: Whether to display progress bars during processing.

    Returns:
        None: Writes the dataset directly to disk.
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
        tar_path: str,
        gen_func: Callable[..., Generator[Sample, None, None]],
        var_features_keys: list[str],
        batch: list[IndexType],
        start_index: int,
        queue: mp.Queue,
    ) -> None:  # pragma: no cover
        """Processes a single batch and writes samples to tar.

        Args:
            tar_path: Path to the tar file for the split.
            gen_func: Generator function for samples.
            var_features_keys: List of feature keys.
            batch: Batch of sample IDs.
            start_index: Starting sample index.
            queue: Queue for progress tracking.
        """
        # Create tar writer for this batch (will be appended to main tar)
        temp_tar = tar_path.replace(".tar", f"_batch_{start_index}.tar")
        with wds.TarWriter(temp_tar) as tar_writer:
            sample_counter = start_index

            for sample in gen_func([batch]):
                _write_sample_to_tar(
                    tar_writer, sample, var_features_keys, sample_counter
                )
                sample_counter += 1
                queue.put(1)

    def tqdm_updater(
        total: int, queue: mp.Queue, desc: str = "Processing"
    ) -> None:  # pragma: no cover
        """Tqdm process that listens to the queue to update progress.

        Args:
            total: Total number of items to process.
            queue: Queue to receive progress updates.
            desc: Description for the progress bar.
        """
        with tqdm(total=total, desc=desc, disable=not verbose) as pbar:
            finished = 0
            while finished < total:
                finished += queue.get()
                pbar.update(1)

    for split_name, gen_func in generators.items():
        tar_path = str(output_folder / f"{split_name}.tar")

        gen_kwargs_ = gen_kwargs or {sn: {} for sn in generators.keys()}
        batch_ids_list = gen_kwargs_.get(split_name, {}).get("shards_ids", [])

        total_samples = sum(len(batch) for batch in batch_ids_list) if batch_ids_list else 0

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
            temp_tars = []

            for batch in batch_ids_list:
                temp_tar = tar_path.replace(".tar", f"_batch_{start_index}.tar")
                temp_tars.append(temp_tar)
                p = mp.Process(
                    target=worker_batch,
                    args=(
                        tar_path,
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

            # Merge temporary tar files
            with wds.TarWriter(tar_path) as main_tar:
                for temp_tar in temp_tars:
                    if Path(temp_tar).exists():
                        with wds.ShardList([temp_tar]) as shard:
                            for sample in shard:
                                main_tar.write(sample)
                        Path(temp_tar).unlink()

        else:
            # Sequential execution
            sample_counter = 0

            # Determine total for progress bar
            if not batch_ids_list:
                # No batch info, estimate or skip progress
                total_samples = None

            with wds.TarWriter(tar_path) as tar_writer:
                with tqdm(
                    total=total_samples,
                    desc=f"Writing {split_name} split",
                    disable=not verbose,
                ) as pbar:
                    for sample in gen_func():
                        _write_sample_to_tar(
                            tar_writer, sample, var_features_keys, sample_counter
                        )
                        sample_counter += 1
                        if total_samples is not None:
                            pbar.update(1)


def push_local_datasetdict_to_hub(
    repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
) -> None:  # pragma: no cover
    """Pushes a local dataset directory to Hugging Face Hub.

    This function uploads the contents of a local directory to a specified
    Hugging Face repository as a dataset. It uses the HfApi to handle large
    folder uploads with configurable parallelism.

    Args:
        repo_id: The Hugging Face repository ID where the dataset will be uploaded.
        local_dir: Path to the local directory containing the dataset files to upload.
        num_workers: Number of worker threads to use for uploading. Defaults to 1.

    Returns:
        None: Uploads the dataset directly to Hugging Face Hub.
    """
    api = HfApi()
    api.upload_large_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=num_workers,
        ignore_patterns=["*.tmp"],
        allow_patterns=["data/*.tar", "*.yaml", "*.yml", "*.json", "README.md"],
    )


def configure_dataset_card(
    repo_id: str,
    infos: dict[str, dict[str, str]],
    local_dir: Union[str, Path],
    viewer: Optional[bool] = None,  # noqa: ARG001
    pretty_name: Optional[str] = None,
    dataset_long_description: Optional[str] = None,
    illustration_urls: Optional[list[str]] = None,
    arxiv_paper_urls: Optional[list[str]] = None,
) -> None:  # pragma: no cover
    """Configures and pushes a dataset card to Hugging Face Hub for a WebDataset backend.

    This function generates a dataset card in YAML format with metadata, features,
    splits information, and usage examples. It automatically detects splits and
    sample counts from the local directory structure, then pushes the card to
    the specified Hugging Face repository.

    Args:
        repo_id: The Hugging Face repository ID where the dataset card will be pushed.
        infos: Dictionary containing dataset metadata, including legal information.
        local_dir: Path to the local directory containing the dataset files.
        viewer: Unused parameter for viewer configuration.
        pretty_name: A human-readable name for the dataset.
        dataset_long_description: A detailed description of the dataset.
        illustration_urls: List of URLs to images that illustrate the dataset.
        arxiv_paper_urls: List of arXiv URLs for papers related to the dataset.

    Returns:
        None: Pushes the dataset card directly to Hugging Face Hub.
    """
    dataset_card_str = """---
task_categories:
- graph-ml
tags:
- physics learning
- geometry learning
---
"""
    local_folder = Path(local_dir)

    # Detect tar files in data directory
    data_dir = local_folder / "data"
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    tar_files = list(data_dir.glob("*.tar"))
    split_names = [f.stem for f in tar_files]

    # Count samples and compute sizes
    nbe_samples = {}
    num_bytes = {}
    size_bytes = 0

    for tar_file in tar_files:
        split_name = tar_file.stem

        # Count samples in tar
        sample_count = 0
        with wds.WebDataset(str(tar_file)) as dataset:
            for _ in dataset:
                sample_count += 1

        nbe_samples[split_name] = sample_count
        num_bytes[split_name] = tar_file.stat().st_size
        size_bytes += num_bytes[split_name]

    lines = dataset_card_str.splitlines()
    lines = [s for s in lines if not s.startswith("license")]

    indices = [i for i, line in enumerate(lines) if line.strip() == "---"]

    assert len(indices) >= 2, (
        "Cannot find two instances of '---', dataset card format error."
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
        lines.insert(count, f"    path: data/{sn}.tar")
        count += 1

    str__ = "\n".join(lines) + "\n"

    if illustration_urls:
        str__ += "<p align='center'>\n"
        for url in illustration_urls:
            str__ += f"<img src='{url}' alt='{url}' width='1000'/>\n"
        str__ += "</p>\n\n"

    str__ += f"```yaml\n{yaml.dump(infos, sort_keys=False, allow_unicode=True)}\n```"

    str__ += """
This dataset was generated with [`plaid`](https://plaid-lib.readthedocs.io/) using the WebDataset backend,
we refer to this documentation for additional details on how to extract data from `plaid_sample` objects.

The simplest way to use this dataset is to first download it:
```python
from plaid.storage import download_from_hub

repo_id = "channel/dataset"
local_folder = "downloaded_dataset"

download_from_hub(repo_id, local_folder, backend="webdataset")
```

Then, to iterate over the dataset and instantiate samples:
```python
from plaid.storage import init_from_disk

local_folder = "downloaded_dataset"
split_name = "train"

datasetdict, converterdict = init_from_disk(local_folder, backend="webdataset")

dataset = datasetdict[split_name]
converter = converterdict[split_name]

for i in range(len(dataset)):
    plaid_sample = converter.to_plaid(dataset, i)
```

It is possible to stream the data directly:
```python
from plaid.storage import init_streaming_from_hub

repo_id = "channel/dataset"

datasetdict, converterdict = init_streaming_from_hub(repo_id, backend="webdataset")

dataset = datasetdict[split_name]
converter = converterdict[split_name]

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
