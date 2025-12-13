import logging
from pathlib import Path
from typing import Optional, Union, Callable, Generator, Optional

import datasets
import yaml
from huggingface_hub import DatasetCard, hf_hub_download

from plaid.storage.hf_datasets.bridge import generator_to_datasetdict
from plaid.storage.hf_datasets.reader import init_datasetdict_from_disk
from plaid import Sample
from plaid.types import IndexType

logger = logging.getLogger(__name__)


def _compute_num_shards(hf_dataset_dict: datasets.DatasetDict) -> dict[str, int]:
    target_shard_size_mb = 500

    num_shards = {}
    for split_name, ds in hf_dataset_dict.items():
        n_samples = len(ds)
        assert n_samples > 0, f"split {split_name} has no sample"

        dataset_size_bytes = ds.data.nbytes
        target_shard_size_bytes = target_shard_size_mb * 1024 * 1024

        n_shards = max(
            1,
            (dataset_size_bytes + target_shard_size_bytes - 1)
            // target_shard_size_bytes,
        )
        num_shards[split_name] = min(n_samples, int(n_shards))
    return num_shards


def save_datasetdict_to_disk(
    path: Union[str, Path], hf_datasetdict: datasets.DatasetDict, **kwargs
) -> None:
    """Save a Hugging Face DatasetDict to disk.

    This function serializes the provided DatasetDict and writes it to the specified
    directory, preserving its features, splits, and data for later loading.

    Args:
        path (Union[str, Path]): Directory path where the DatasetDict will be saved.
        hf_dataset_dict (datasets.DatasetDict): The Hugging Face DatasetDict to save.
        **kwargs:
            Keyword arguments forwarded to
            [`DatasetDict.save_to_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.save_to_disk).

    Returns:
        None
    """
    num_shards = _compute_num_shards(hf_datasetdict)
    num_proc = kwargs.get("num_proc", None)
    if num_proc is not None:  # pragma: no cover
        min_num_shards = min(num_shards.values())
        if min_num_shards < num_proc:
            logger.warning(
                f"num_proc chaged from {num_proc} to 1 to safely adapt for num_shards={num_shards}"
            )
            num_proc = 1
        del kwargs["num_proc"]

    hf_datasetdict.save_to_disk(
        str(Path(path) / "data"), num_shards=num_shards, num_proc=num_proc, **kwargs
    )


def generate_datasetdict_to_disk(
    output_folder: Union[str, Path],
    generators: dict[str, Callable[..., Generator[Sample, None, None]]],
    variable_schema: dict[str, dict],
    gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
    num_proc: int = 1,
    verbose: bool = False,  # noqa: ARG001
) -> None:

    hf_datasetdict = generator_to_datasetdict(
        generators,
        variable_schema,
        gen_kwargs=gen_kwargs,
        processes_number=num_proc,
    )
    save_datasetdict_to_disk(output_folder, hf_datasetdict, num_proc=num_proc)


def push_datasetdict_to_hub(
    repo_id: str, hf_datasetdict: datasets.DatasetDict, **kwargs
) -> None:  # pragma: no cover (not tested in unit tests)
    """Push a Hugging Face `DatasetDict` to the Hugging Face Hub.

    This is a thin wrapper around `datasets.DatasetDict.push_to_hub`, allowing
    you to upload a dataset dictionary (with one or more splits such as
    `"train"`, `"validation"`, `"test"`) to the Hugging Face Hub.

    Note:
        The function automatically handles sharding of the dataset by setting `num_shards`
        for each split. For each split, the number of shards is set to the minimum between
        the number of samples in that split and such that shards are targetted to approx. 500 MB.
        This ensures efficient chunking while preventing excessive fragmentation. Empty splits
        will raise an assertion error.

    Args:
        repo_id (str):
            The repository ID on the Hugging Face Hub
            (e.g. `"username/dataset_name"`).
        hf_dataset_dict (datasets.DatasetDict):
            The Hugging Face dataset dictionary to push.
        **kwargs:
            Keyword arguments forwarded to
            [`DatasetDict.push_to_hub`](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.DatasetDict.push_to_hub).

    Returns:
        None
    """
    num_shards = _compute_num_shards(hf_datasetdict)
    num_proc = kwargs.get("num_proc", None)
    if num_proc is not None:  # pragma: no cover
        min_num_shards = min(num_shards.values())
        if min_num_shards < num_proc:
            logger.warning(
                f"num_proc chaged from {num_proc} to 1 to safely adapt for num_shards={num_shards}"
            )
            num_proc = 1
        del kwargs["num_proc"]

    hf_datasetdict.push_to_hub(
        repo_id, num_shards=num_shards, num_proc=num_proc, **kwargs
    )


def push_local_datasetdict_to_hub(repo_id, local_dir, num_workers=1):
    datasetdict = init_datasetdict_from_disk(local_dir)
    push_datasetdict_to_hub(repo_id, datasetdict, num_proc=num_workers)


def configure_dataset_card(
    repo_id: str,
    infos: dict[str, dict[str, str]],
    local_dir: Optional[Union[str, Path]] = None,  # noqa: ARG001
    variable_schema: Optional[dict] = None,  # noqa: ARG001
    viewer: bool = False,
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
    readme_path = hf_hub_download(
        repo_id=repo_id, filename="README.md", repo_type="dataset"
    )

    with open(readme_path, "r", encoding="utf-8") as f:
        dataset_card_str = f.read()

    lines = dataset_card_str.splitlines()
    lines = [s for s in lines if not s.startswith("license")]

    indices = [i for i, line in enumerate(lines) if line.strip() == "---"]

    assert len(indices) >= 2, (
        "Cannot find two instances of '---', you should try to update a correct dataset_card."
    )
    lines = lines[: indices[1] + 1]

    count = 1
    lines.insert(count, f"license: {infos['legal']['license']}")
    count += 1
    if viewer is False:
        lines.insert(count, "viewer: false")
        count += 1
    lines.insert(count, "task_categories:")
    count += 1
    lines.insert(count, "- graph-ml")
    count += 1
    if pretty_name:
        lines.insert(count, f"pretty_name: {pretty_name}")
        count += 1
    lines.insert(count, "tags:")
    count += 1
    lines.insert(count, "- physics learning")
    count += 1
    lines.insert(count, "- geometry learning")
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

For those familiar with HF's `datasets` library, raw data can be retrieved without using the `plaid` library:
```python
from datasets import load_dataset

repo_id = "channel/dataset"

datasetdict = load_dataset(repo_id)

for split_name, dataset in datasetdict.items():
    for raw_sample in dataset:
        for feat_name in dataset.column_names:
            feature = raw_sample[feat_name]
```
Notice that raw data refers to the variable features only, with a specific encoding for time variable features.
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
