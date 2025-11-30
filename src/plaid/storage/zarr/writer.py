import multiprocessing as mp
from pathlib import Path
from typing import Callable, Generator, Optional, Union

import numpy as np
import yaml
import zarr
from huggingface_hub import DatasetCard, HfApi
from tqdm import tqdm

from plaid import Sample
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.types import IndexType


def flatten_path(key: str) -> str:
    return key.replace("/", "__")


def auto_chunks(shape, target_n):
    # ensure pure Python ints
    target_n = int(target_n)
    shape = tuple(int(s) for s in shape)

    # elements in one "row"
    elems_per_slice = int(np.prod(shape[1:]) or 1)

    rows = max(1, target_n // elems_per_slice)
    rows = min(rows, shape[0])  # cannot exceed the dimension size

    return (rows,) + shape[1:]


def save_datasetdict_to_disk(
    output_folder: Union[str, Path],
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    variable_schema: dict[str, dict],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    num_proc: int = 1,
    verbose: bool = False,
) -> None:
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
        split_root_path, gen_func, var_features_keys, batch, start_index, queue
    ):
        """Process a single batch and write samples to Zarr."""
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
                    flatten_path(key),
                    data=value,
                    chunks=auto_chunks(value.shape, 5_000_000),
                )  # chunks=value.shape

            sample_counter += 1
            queue.put(1)

    def tqdm_updater(total, queue, desc="Processing"):
        """Tqdm process that listens to the queue to update progress."""
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

        if num_proc > 1 and batch_ids_list:
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
                        g.create_array(
                            flatten_path(key),
                            data=value,
                            chunks=auto_chunks(value.shape, 5_000_000),
                        )  # chunks=value.shape

                    sample_counter += 1
                    pbar.update(1)


def push_datasetdict_to_hub(repo_id, local_dir, num_workers=1):
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
    local_folder: Union[str, Path],
    infos: dict[str, dict[str, str]],
    variable_schema: Optional[dict] = None,
    pretty_name: Optional[str] = None,
    dataset_long_description: Optional[str] = None,
    illustration_urls: Optional[list[str]] = None,
    arxiv_paper_urls: Optional[list[str]] = None,
) -> None:
    r"""Update a dataset card with PLAID-specific metadata and documentation.

    Args:
        dataset_card (str): The original dataset card content to update.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset information
            with "legal" and "data_production" sections. Defaults to None.
        pretty_name (str, optional): A human-readable name for the dataset. Defaults to None.
        dataset_long_description (str, optional): Detailed description of the dataset's content,
            purpose, and characteristics. Defaults to None.
        illustration_urls (list[str], optional): List of URLs to images illustrating the dataset.
            Defaults to None.
        arxiv_paper_urls (list[str], optional): List of URLs to related arXiv papers.
            Defaults to None.

    Returns:
        str: The updated dataset card content as a string.

    Example:
        ```python
        # Create initial dataset card
        card = "---\ndataset_name: my_dataset\n---"

        # Update with PLAID-specific content
        updated_card = update_dataset_card(
            dataset_card=card,
            license="mit",
            pretty_name="My PLAID Dataset",
            dataset_long_description="This dataset contains...",
            illustration_urls=["https://example.com/image.png"],
            arxiv_paper_urls=["https://arxiv.org/abs/..."]
        )

        # Push to Hugging Face Hub
        from huggingface_hub import DatasetCard
        dataset_card = DatasetCard(updated_card)
        dataset_card.push_to_hub("username/dataset")
        ```
    """

    def _dict_to_list_format(d: dict) -> str:
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
    local_folder = Path(local_folder)
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
    if pretty_name:
        lines.insert(count, f"pretty_name: {pretty_name}")
        count += 1

    lines.insert(count, "dataset_info:")
    count += 1
    if variable_schema is not None:
        lines.insert(count, "  features:")
        count += 1
        for fn, type_ in variable_schema.items():
            lines.insert(count, f"  - name: {flatten_path(fn)}")
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
Example of commands [TO UPDATE FOR ZARR]:
```python
from datasets import load_dataset
from plaid.bridges import huggingface_bridge

repo_id = "chanel/dataset"
pb_def_name = "pb_def_name" #`pb_def_name` is to choose from the repo `problem_definitions` folder

# Load the dataset
hf_datasetdict = load_dataset(repo_id)

# Load addition required data
flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, pb_def_name)

# Efficient reconstruction of plaid samples
for split_name, hf_dataset in hf_datasetdict.items():
    for i in range(len(hf_dataset)):
        sample = huggingface_bridge.to_plaid_sample(
            hf_dataset,
            i,
            flat_cst[split_name],
            key_mappings["cgns_types"],
        )

# Extract input and output features from samples:
for t in sample.get_all_mesh_times():
    for path in pb_def.get_in_features_identifiers():
        sample.get_feature_by_path(path=path, time=t)
    for path in pb_def.get_out_features_identifiers():
        sample.get_feature_by_path(path=path, time=t)
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
