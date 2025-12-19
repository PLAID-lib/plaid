# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""CGNS dataset writer module.

This module provides functionality for writing datasets in CGNS format for the PLAID library.
It includes utilities for generating datasets from sample generators, saving to disk,
uploading to Hugging Face Hub, and configuring dataset cards.
"""

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Callable, Generator, Optional, Union

import yaml
from huggingface_hub import DatasetCard, HfApi
from tqdm import tqdm

from plaid import Sample
from plaid.types import IndexType

logger = logging.getLogger(__name__)


def generate_datasetdict_to_disk(
    output_folder: Union[str, Path],
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    variable_schema: Optional[dict[str, dict]] = None,  # noqa: ARG001
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    num_proc: int = 1,
    verbose: bool = False,
) -> None:
    """Generates and saves a dataset to disk in CGNS format.

    Args:
        output_folder: Base directory to save the dataset.
        generators: Dict of split generators.
        variable_schema: Unused variable schema.
        gen_kwargs: Optional generator kwargs for parallel processing.
        num_proc: Number of processes.
        verbose: Whether to show progress.
    """
    output_folder = Path(output_folder)

    assert (gen_kwargs is None and num_proc == 1) or (
        gen_kwargs is not None and num_proc > 1
    ), (
        "Invalid configuration: either provide only `generators` with "
        "`num_proc == 1`, or provide `gen_kwargs` with "
        "`num_proc > 1`."
    )

    output_folder = output_folder / "data"
    output_folder.mkdir(exist_ok=True, parents=True)

    def worker_batch(
        gen_func: Callable, batch: list[IndexType], start_index: int, queue: mp.Queue
    ) -> None:  # pragma: no cover
        """Process a single batch and write samples to disk."""
        sample_counter = start_index

        for sample in gen_func([batch]):
            sample.save_to_dir(split_path / f"sample_{sample_counter:09d}")

            sample_counter += 1
            queue.put(1)

    def tqdm_updater(
        total: int, queue: mp.Queue, desc: str = "Processing"
    ) -> None:  # pragma: no cover
        """Tqdm process that listens to the queue to update progress."""
        with tqdm(total=total, desc=desc, disable=not verbose) as pbar:
            finished = 0
            while finished < total:
                finished += queue.get()
                pbar.update(1)

    for split_name, gen_func in generators.items():
        split_path = output_folder / split_name
        split_path.mkdir(exist_ok=True, parents=True)

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
                        gen_func,
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
                    sample.save_to_dir(split_path / f"sample_{sample_counter:09d}")

                    sample_counter += 1
                    pbar.update(1)


def push_local_datasetdict_to_hub(
    repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
) -> None:  # pragma: no cover
    """Pushes a local dataset directory to Hugging Face Hub.

    Args:
        repo_id: The repository ID.
        local_dir: Local directory path.
        num_workers: Number of upload workers.
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
    """Configures and pushes a dataset card to Hugging Face Hub for a CGNS backend dataset.

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

    def _dict_to_list_format(d: dict) -> str:
        """Converts a variable schema dict to YAML list format."""
        dtype = d.get("dtype", "unknown")
        ndim = d.get("ndim", 1)

        lines = []
        current_indent = "    "

        for i in range(ndim - 1):
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
            lines.insert(count, f"  - name: {fn}")
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
    str__ += "This dataset was generated in [PLAID](https://plaid-lib.readthedocs.io/), we refer to this documentation for additional details on how to extract data from `sample` objects.\n"

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
