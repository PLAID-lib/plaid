"""Hugging Face bridge for PLAID datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
import pickle
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")


import datasets
from datasets import load_dataset

from plaid import Dataset, ProblemDefinition, Sample
from plaid.bridges._huggingface_helpers import (
    _HFShardToPlaidSampleConverter,
    _HFToPlaidSampleConverter,
)
from plaid.types import IndexType

"""
Convention with hf (Hugging Face) datasets:
- hf-datasets contains a single Hugging Face split, named 'all_samples'.
- samples contains a single Hugging Face feature, named called "sample".
- Samples are instances of plaid.containers.sample.Sample.
- Mesh objects included in samples follow the CGNS standard, and can be converted in Muscat.Containers.Mesh.Mesh.
- problem_definition info is stored in hf-datasets "description" parameter
"""


def generate_huggingface_description(
    infos: dict, problem_definition: ProblemDefinition
) -> dict[str, Any]:
    """Generates a Hugging Face dataset description field from a plaid dataset infos and problem definition.

    The conventions chosen here ensure working conversion to and from huggingset datasets.

    Args:
        infos (dict): infos entry of the plaid dataset from which the Hugging Face description is to be generated
        problem_definition (ProblemDefinition): of which the Hugging Face description is to be generated

    Returns:
        dict[str]: Hugging Face dataset description
    """
    # type hinting the values as Any because they can be of various types
    description: dict[str, Any] = {}

    description.update(infos)

    split: dict[str, IndexType] = problem_definition.get_split(indices_name=None)  # pyright: ignore[reportAssignmentType]
    description["split"] = split
    description["task"] = problem_definition.get_task()

    description["in_scalars_names"] = problem_definition.in_scalars_names
    description["out_scalars_names"] = problem_definition.out_scalars_names
    description["in_timeseries_names"] = problem_definition.in_timeseries_names
    description["out_timeseries_names"] = problem_definition.out_timeseries_names
    description["in_fields_names"] = problem_definition.in_fields_names
    description["out_fields_names"] = problem_definition.out_fields_names
    description["in_meshes_names"] = problem_definition.in_meshes_names
    description["out_meshes_names"] = problem_definition.out_meshes_names
    return description


def plaid_dataset_to_huggingface(
    dataset: Dataset,
    problem_definition: ProblemDefinition,
    split: str = "all_samples",
    processes_number: int = 1,
) -> datasets.Dataset:
    """Use this function for converting a Hugging Face dataset from a plaid dataset.

    The dataset can then be saved to disk, or pushed to the Hugging Face hub.

    Args:
        dataset (Dataset): the plaid dataset to be converted in Hugging Face format
        problem_definition (ProblemDefinition): the problem definition is used to generate the description of the Hugging Face dataset.
        split (str): The name of the split. Default: "all_samples".
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.Dataset: dataset in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_dataset_to_huggingface(dataset, problem_definition, split)
            dataset.save_to_disk("path/to/dir)
            dataset.push_to_hub("chanel/dataset")
    """
    if split == "all_samples":
        ids = dataset.get_sample_ids()
    else:
        ids = problem_definition.get_split(split)

    def generator():
        for sample in dataset[ids]:
            yield {
                "sample": pickle.dumps(sample.model_dump()),
            }

    return plaid_generator_to_huggingface(
        generator=generator,
        infos=dataset.get_infos(),
        problem_definition=problem_definition,
        split=split,
        processes_number=processes_number,
    )


def plaid_dataset_to_huggingface_datasetdict(
    dataset: Dataset,
    problem_definition: ProblemDefinition,
    main_splits: list[str],
    processes_number: int = 1,
) -> datasets.DatasetDict:
    """Use this function for converting a Hugging Face dataset dict from a plaid dataset.

    The dataset can then be saved to disk, or pushed to the Hugging Face hub.

    Args:
        dataset (Dataset): the plaid dataset to be converted in Hugging Face format
        problem_definition (ProblemDefinition): the problem definition is used to generate the description of the Hugging Face dataset.
        main_splits (list[str]): The name of the main splits: defining a partitioning of the sample ids.
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.Dataset: dataset in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_dataset_to_huggingface(dataset, problem_definition, split)
            dataset.save_to_disk("path/to/dir)
            dataset.push_to_hub("chanel/dataset")
    """
    _dict = {}
    for _, split in enumerate(main_splits):
        ds = plaid_dataset_to_huggingface(
            dataset=dataset,
            problem_definition=problem_definition,
            split=split,
            processes_number=processes_number,
        )
        _dict[split] = ds

    return datasets.DatasetDict(_dict)


def plaid_generator_to_huggingface(
    generator: Callable,
    infos: dict,
    problem_definition: ProblemDefinition,
    split: str = "all_samples",
    processes_number: int = 1,
) -> datasets.Dataset:
    """Use this function for creating a Hugging Face dataset from a sample generator function.

    This function can be used when the plaid dataset cannot be loaded in RAM all at once due to its size.
    The generator enables loading samples one by one.
    The dataset can then be saved to disk, or pushed to the Hugging Face hub.

    Args:
        generator (Callable): a function yielding a dict {"sample" : sample}, where sample is of type 'bytes'
        infos (dict):  the info is used to generate the description of the Hugging Face dataset.
        problem_definition (ProblemDefinition): the problem definition is used to generate the description of the Hugging Face dataset.
        split (str): The name of the split. Default: "all_samples".
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.Dataset: dataset in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_generator_to_huggingface(generator, infos, split, problem_definition)
            dataset.push_to_hub("chanel/dataset")
            dataset.save_to_disk("path/to/dir")
    """
    ds: datasets.Dataset = datasets.Dataset.from_generator(  # pyright: ignore[reportAssignmentType]
        generator,
        features=datasets.Features({"sample": datasets.Value("binary")}),
        num_proc=processes_number,
        writer_batch_size=1,
        split=datasets.splits.NamedSplit(split),
    )

    def update_dataset_description(
        ds: datasets.Dataset, new_desc: dict[str, Any]
    ) -> datasets.Dataset:
        info = ds.info.copy()
        info.description = new_desc  # pyright: ignore[reportAttributeAccessIssue] -> info.description is HF's DatasetInfo. We might want to correct this later.
        ds._info = info
        return ds

    new_description: dict[str, Any] = generate_huggingface_description(
        infos, problem_definition
    )
    ds = update_dataset_description(ds, new_description)

    return ds


def plaid_generator_to_huggingface_datasetdict(
    generator: Callable,
    infos: dict,
    problem_definition: ProblemDefinition,
    main_splits: list,
    processes_number: int = 1,
) -> datasets.DatasetDict:
    """Use this function for creating a Hugging Face dataset dict (containing multiple splits) from a sample generator function.

    This function can be used when the plaid dataset cannot be loaded in RAM all at once due to its size.
    The generator enables loading samples one by one.
    The dataset dict can then be saved to disk, or pushed to the Hugging Face hub.

    Notes:
        Only the first split will contain the decription.

    Args:
        generator (Callable): a function yielding a dict {"sample" : sample}, where sample is of type 'bytes'
        infos (dict): infos entry of the plaid dataset from which the Hugging Face dataset is to be generated
        problem_definition (ProblemDefinition): the problem definition is used to generate the description of the Hugging Face dataset.
        main_splits (str, optional): The name of the main splits: defining a partitioning of the sample ids.
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.DatasetDict: dataset dict in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_generator_to_huggingface_datasetdict(generator, infos, problem_definition, main_splits)
            dataset.push_to_hub("chanel/dataset")
            dataset.save_to_disk("path/to/dir")
    """
    _dict = {}
    for _, split in enumerate(main_splits):
        ds = plaid_generator_to_huggingface(
            generator,
            infos,
            problem_definition=problem_definition,
            split=split,
            processes_number=processes_number,
        )
        _dict[split] = ds

    return datasets.DatasetDict(_dict)


def huggingface_description_to_problem_definition(
    description: dict,
) -> ProblemDefinition:
    """Converts a Hugging Face dataset description to a plaid problem definition.

    Args:
        description (dict): the description field of a Hugging Face dataset, containing the problem definition

    Returns:
        problem_definition (ProblemDefinition): the plaid problem definition initialized from the Hugging Face dataset description
    """
    problem_definition = ProblemDefinition()
    problem_definition.set_task(description["task"])
    problem_definition.set_split(description["split"])
    problem_definition.add_input_scalars_names(description["in_scalars_names"])
    problem_definition.add_output_scalars_names(description["out_scalars_names"])
    problem_definition.add_input_timeseries_names(description["in_timeseries_names"])
    problem_definition.add_output_timeseries_names(description["out_timeseries_names"])
    problem_definition.add_input_fields_names(description["in_fields_names"])
    problem_definition.add_output_fields_names(description["out_fields_names"])
    problem_definition.add_input_meshes_names(description["in_meshes_names"])
    problem_definition.add_output_meshes_names(description["out_meshes_names"])

    return problem_definition


def to_plaid_sample(hf_sample: dict[str, Any]) -> Sample:
    """Convert a Hugging Face sample dictionary to a PLAID Sample instance.

    Args:
        hf_sample (dict[str, Any]): A dictionary with a "sample" key containing the pickled sample bytes.

    Returns:
        Sample: The deserialized PLAID Sample object.
    """
    return Sample.model_validate(pickle.loads(hf_sample["sample"]))


def huggingface_dataset_to_plaid(
    ds: datasets.Dataset,
    ids: Optional[list[int]] = None,
    processes_number: int = 1,
    large_dataset: bool = False,
    verbose: bool = True,
) -> tuple[Dataset, ProblemDefinition]:
    """Use this function for converting a plaid dataset from a Hugging Face dataset.

    A Hugging Face dataset can be read from disk or the hub. From the hub, the
    split = "all_samples" options is important to get a dataset and not a datasetdict.
    Many options from loading are available (caching, streaming, etc...)

    Args:
        ds (datasets.Dataset): the dataset in Hugging Face format to be converted
        ids (list, optional): The specific sample IDs to load from the dataset. Defaults to None.
        processes_number (int, optional): The number of processes used to generate the plaid dataset
        large_dataset (bool): if True, uses a variant where parallel worker do not each load the complete dataset. Default: False.
        verbose (bool, optional): if True, prints progress using tdqm

    Returns:
        dataset (Dataset): the converted dataset.
        problem_definition (ProblemDefinition): the problem definition generated from the Hugging Face dataset

    Example:
        .. code-block:: python

            from datasets import load_dataset, load_from_disk

            dataset = load_dataset("path/to/dir", split = "all_samples")
            dataset = load_from_disk("chanel/dataset")
            plaid_dataset, plaid_problem = huggingface_dataset_to_plaid(dataset)
    """
    assert processes_number <= len(ds), (
        "Trying to parallelize with more processes than samples in dataset"
    )
    if ids:
        assert processes_number <= len(ids), (
            "Trying to parallelize with more processes than selected samples in dataset"
        )

    dataset = Dataset()

    if verbose:
        print("Converting Hugging Face dataset to plaid dataset...")

    if large_dataset:
        if ids:
            raise NotImplementedError(
                "ids selection not implemented with large_dataset option"
            )
        for i in range(processes_number):
            shard = ds.shard(num_shards=processes_number, index=i)
            shard.save_to_disk(f"./shards/dataset_shard_{i}")

        def parallel_convert(shard_path, n_workers):
            converter = _HFShardToPlaidSampleConverter(shard_path)
            with Pool(processes=n_workers) as pool:
                return list(
                    tqdm(
                        pool.imap(converter, range(len(converter.ds))),
                        total=len(converter.ds),
                        disable=not verbose,
                    )
                )

        samples = []

        for i in range(processes_number):
            shard_path = Path(".") / "shards" / f"dataset_shard_{i}"
            shard_samples = parallel_convert(shard_path, n_workers=processes_number)
            samples.extend(shard_samples)

        dataset.add_samples(samples, ids)

        shards_dir = Path(".") / "shards"
        if shards_dir.exists() and shards_dir.is_dir():
            shutil.rmtree(shards_dir)

    else:
        if ids:
            indices = ids
        else:
            indices = range(len(ds))

        with Pool(processes=processes_number) as pool:
            for idx, sample in enumerate(
                tqdm(
                    pool.imap(_HFToPlaidSampleConverter(ds), indices),
                    total=len(indices),
                    disable=not verbose,
                )
            ):
                dataset.add_sample(sample, id=indices[idx])

    infos = {}
    if "legal" in ds.description:
        infos["legal"] = ds.description["legal"]
    if "data_production" in ds.description:
        infos["data_production"] = ds.description["data_production"]

    dataset.set_infos(infos)

    problem_definition = huggingface_description_to_problem_definition(ds.description)

    return dataset, problem_definition


def streamed_huggingface_dataset_to_plaid(
    hf_repo: str,
    number_of_samples: int,
) -> tuple[Dataset, ProblemDefinition]:  # pragma: no cover
    """Use this function for creating a plaid dataset by streaming on Hugging Face.

    The indices of the retrieved sample is not controled.

    Args:
        hf_repo (str): the name of the repo on Hugging Face
        number_of_samples (int): The number of samples to retrieve.

    Returns:
        dataset (Dataset): the converted dataset.
        problem_definition (ProblemDefinition): the problem definition generated from the Hugging Face dataset

    Notes:
        .. code-block:: python

            from plaid.bridges.huggingface_bridge import streamed_huggingface_dataset_to_plaid

            dataset, pb_def = streamed_huggingface_dataset_to_plaid('PLAID-datasets/VKI-LS59', 2)
    """
    ds_stream = load_dataset(hf_repo, split="all_samples", streaming=True)

    samples = []
    for _ in range(number_of_samples):
        hf_sample = next(iter(ds_stream))
        samples.append(to_plaid_sample(hf_sample))

    dataset = Dataset.from_list_of_samples(samples)

    infos = {}
    if "legal" in ds_stream.description:
        infos["legal"] = ds_stream.description["legal"]
    if "data_production" in ds_stream.description:
        infos["data_production"] = ds_stream.description["data_production"]

    dataset.set_infos(infos)

    problem_definition = huggingface_description_to_problem_definition(
        ds_stream.description
    )

    return dataset, problem_definition


def create_string_for_huggingface_dataset_card(
    description: dict,
    download_size_bytes: int,
    dataset_size_bytes: int,
    nb_samples: int,
    owner: str,
    license: str,
    zenodo_url: Optional[str] = None,
    arxiv_paper_url: Optional[str] = None,
    pretty_name: Optional[str] = None,
    size_categories: Optional[list[str]] = None,
    task_categories: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    dataset_long_description: Optional[str] = None,
    url_illustration: Optional[str] = None,
) -> str:
    """Use this function for creating a dataset card, to upload together with the datase on the Hugging Face hub.

    Doing so ensure that load_dataset from the hub will populate the hf-dataset.description field, and be compatible for conversion to plaid.

    Without a dataset_card, the description field is lost.

    The parameters download_size_bytes and dataset_size_bytes can be determined after a
    dataset has been uploaded on Hugging Face:
    - manually by reading their values on the dataset page README.md,
    - automatically as shown in the example below

    See `the hugginface examples <https://github.com/PLAID-lib/plaid/blob/main/examples/bridges/huggingface_bridge_example.py>`__ for a concrete use.

    Args:
        description (dict): Hugging Face dataset description. Obtained from
        - description = hf_dataset.description
        - description = generate_huggingface_description(infos, problem_definition)
        download_size_bytes (int): the size of the dataset when downloaded from the hub
        dataset_size_bytes (int): the size of the dataset when loaded in RAM
        nb_samples (int): the number of samples in the dataset
        owner (str): the owner of the dataset, usually a username or organization name on Hugging Face
        license (str): the license of the dataset, e.g. "CC-BY-4.0", "CC0-1.0", etc.
        zenodo_url (str, optional): the Zenodo URL of the dataset, if available
        arxiv_paper_url (str, optional): the arxiv paper URL of the dataset, if available
        pretty_name (str, optional): a human-readable name for the dataset, e.g. "PLAID Dataset"
        size_categories (list[str], optional): size categories of the dataset, e.g. ["small", "medium", "large"]
        task_categories (list[str], optional): task categories of the dataset, e.g. ["image-classification", "text-generation"]
        tags (list[str], optional): tags for the dataset, e.g. ["3D", "simulation", "mesh"]
        dataset_long_description (str, optional): a long description of the dataset, providing more details about its content and purpose
        url_illustration (str, optional): a URL to an illustration image for the dataset, e.g. a screenshot or a sample mesh

    Returns:
        dataset (Dataset): the converted dataset
        problem_definition (ProblemDefinition): the problem definition generated from the Hugging Face dataset

    Example:
        .. code-block:: python

            hf_dataset.push_to_hub("chanel/dataset")

            from datasets import load_dataset_builder

            datasetInfo = load_dataset_builder("chanel/dataset").__getstate__()['info']

            from huggingface_hub import DatasetCard

            card_text = create_string_for_huggingface_dataset_card(
                description = description,
                download_size_bytes = datasetInfo.download_size,
                dataset_size_bytes = datasetInfo.dataset_size,
                ...)
            dataset_card = DatasetCard(card_text)
            dataset_card.push_to_hub("chanel/dataset")
    """
    str__ = f"""---
license: {license}
"""

    if size_categories:
        str__ += f"""size_categories:
  {size_categories}
"""

    if task_categories:
        str__ += f"""task_categories:
  {task_categories}
"""

    if pretty_name:
        str__ += f"""pretty_name: {pretty_name}
"""

    if tags:
        str__ += f"""tags:
  {tags}
"""

    str__ += f"""configs:
  - config_name: default
    data_files:
      - split: all_samples
        path: data/all_samples-*
dataset_info:
  description: {description}
  features:
  - name: sample
    dtype: binary
  splits:
  - name: all_samples
    num_bytes: {dataset_size_bytes}
    num_examples: {nb_samples}
  download_size: {download_size_bytes}
  dataset_size: {dataset_size_bytes}
---

# Dataset Card
"""
    if url_illustration:
        str__ += f"""![image/png]({url_illustration})

This dataset contains a single Hugging Face split, named 'all_samples'.

The samples contains a single Hugging Face feature, named called "sample".

Samples are instances of [plaid.containers.sample.Sample](https://plaid-lib.readthedocs.io/en/latest/autoapi/plaid/containers/sample/index.html#plaid.containers.sample.Sample).
Mesh objects included in samples follow the [CGNS](https://cgns.github.io/) standard, and can be converted in
[Muscat.Containers.Mesh.Mesh](https://muscat.readthedocs.io/en/latest/_source/Muscat.Containers.Mesh.html#Muscat.Containers.Mesh.Mesh).


Example of commands:
```python
import pickle
from datasets import load_dataset
from plaid import Sample

# Load the dataset
dataset = load_dataset("chanel/dataset", split="all_samples")

# Get the first sample of the first split
split_names = list(dataset.description["split"].keys())
ids_split_0 = dataset.description["split"][split_names[0]]
sample_0_split_0 = dataset[ids_split_0[0]]["sample"]
plaid_sample = Sample.model_validate(pickle.loads(sample_0_split_0))
print("type(plaid_sample) =", type(plaid_sample))

print("plaid_sample =", plaid_sample)

# Get a field from the sample
field_names = plaid_sample.get_field_names()
field = plaid_sample.get_field(field_names[0])
print("field_names[0] =", field_names[0])

print("field.shape =", field.shape)

# Get the mesh and convert it to Muscat
from Muscat.Bridges import CGNSBridge
CGNS_tree = plaid_sample.get_mesh()
mesh = CGNSBridge.CGNSToMesh(CGNS_tree)
print(mesh)
```

## Dataset Details

### Dataset Description

"""

    if dataset_long_description:
        str__ += f"""{dataset_long_description}
"""

    str__ += f"""- **Language:** [PLAID](https://plaid-lib.readthedocs.io/)
- **License:** {license}
- **Owner:** {owner}
"""

    if zenodo_url or arxiv_paper_url:
        str__ += """
### Dataset Sources

"""

    if zenodo_url:
        str__ += f"""- **Repository:** [Zenodo]({zenodo_url})
"""

    if arxiv_paper_url:
        str__ += f"""- **Paper:** [arxiv]({arxiv_paper_url})
"""

    return str__
