"""Huggingface bridge for PLAID datasets."""

import pickle
import sys
from typing import Callable

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import datasets

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition

"""
Convention with hf (huggingface) datasets:
- hf-datasets contains a single huggingface split, named 'all_samples'.
- samples contains a single huggingface feature, named called "sample".
- Samples are instances of plaid.containers.sample.Sample.
- Mesh objects included in samples follow the CGNS standard, and can be converted in Muscat.Containers.Mesh.Mesh.
- problem_definition info is stored in hf-datasets "description" parameter
"""


def generate_huggingface_description(
    infos: dict, problem_definition: ProblemDefinition
) -> dict[str]:
    """Generates a huggingface dataset description field from a plaid dataset infos and problem definition.

    The conventions chosen here ensure working conversion to and from huggingset datasets.

    Args:
        infos (dict): infos entry of the plaid dataset from which the huggingface description is to be generated
        problem_definition (ProblemDefinition): of which the huggingface description is to be generated

    Returns:
        dict[str]: huggingface dataset description
    """
    description = {}

    description.update(infos)

    description["split"] = problem_definition.get_split()
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
    dataset: Dataset, problem_definition: ProblemDefinition, processes_number: int = 1
) -> datasets.Dataset:
    """Use this function for converting a huggingface dataset from a plaid dataset.

    The dataset can then be saved to disk, or pushed to the huggingface hub.

    Args:
        dataset (Dataset): the plaid dataset to be converted in huggingface format
        problem_definition (ProblemDefinition): from which the huggingface dataset is to be generated
        processes_number (int, optional): The number of processes used to generate the huggingface dataset

    Returns:
        datasets.Dataset: dataset in huggingface format

    Example:
        .. code-block:: python

            dataset = plaid_dataset_to_huggingface(dataset, problem_definition)
            dataset.save_to_disk("path/to/dir)
            dataset.push_to_hub("chanel/dataset")
    """

    def generator():
        for id in range(len(dataset)):
            yield {
                "sample": pickle.dumps(dataset[id].model_dump()),
            }

    return plaid_generator_to_huggingface(
        generator, dataset.get_infos(), problem_definition, processes_number
    )


def plaid_generator_to_huggingface(
    generator: Callable,
    infos: dict,
    problem_definition: ProblemDefinition,
    processes_number: int = 1,
) -> datasets.Dataset:
    """Use this function for creating a huggingface dataset from a sample generator function.

    This function can be used when the plaid dataset cannot be loaded in RAM all at once due to its size.
    The generator enables loading samples one by one.
    The dataset can then be saved to disk, or pushed to the huggingface hub.

    Args:
        generator (Callable): a function yielding a dict {"sample" : sample}, where sample is of type 'bytes'
        infos (dict): infos entry of the plaid dataset from which the huggingface dataset is to be generated
        problem_definition (ProblemDefinition): from which the huggingface dataset is to be generated
        processes_number (int, optional): The number of processes used to generate the huggingface dataset

    Returns:
        datasets.Dataset: dataset in huggingface format

    Example:
        .. code-block:: python

            dataset = plaid_generator_to_huggingface(generator, infos, problem_definition)
            dataset.push_to_hub("chanel/dataset")
            dataset.save_to_disk("path/to/dir")
    """
    ds = datasets.Dataset.from_generator(
        generator, num_proc=processes_number, writer_batch_size=1
    )

    ds._split = datasets.splits.NamedSplit("all_samples")

    ds._info = datasets.DatasetInfo(
        features=datasets.Features({"sample": datasets.Value("binary")}),
        description=generate_huggingface_description(infos, problem_definition),
    )

    return ds


def huggingface_description_to_problem_definition(
    description: dict,
) -> ProblemDefinition:
    """Docstring to complete."""
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


def huggingface_dataset_to_plaid(
    ds: datasets.Dataset,
) -> tuple[Self, ProblemDefinition]:
    """Use this function for converting a plaid dataset from a huggingface dataset.

    A huggingface dataset can be read from disk or the hub. From the hub, the
    split = "all_samples" options is important to get a dataset and not a datasetdict.
    Many options from loading are available (caching, streaming, etc...)

    Args:
        ds (datasets.Dataset): the dataset in huggingface format to be converted

    Returns:
        dataset (Dataset): the converted dataset.
        problem_definition (ProblemDefinition): the problem definition generated from th huggingface dataset

    Example:
        .. code-block:: python

            from datasets import load_dataset, load_from_disk

            dataset = load_dataset("path/to/dir", split = "all_samples")
            dataset = load_from_disk("chanel/dataset")
            plaid_dataset, plaid_problem = huggingface_dataset_to_plaid(dataset)
    """
    dataset = Dataset()
    for i in range(len(ds)):
        dataset.add_sample(Sample.model_validate(pickle.loads(ds[i]["sample"])))

    infos = {}
    if "legal" in ds.description:
        infos["legal"] = ds.description["legal"]
    if "data_production" in ds.description:
        infos["data_production"] = ds.description["data_production"]

    dataset.set_infos(infos)

    problem_definition = huggingface_description_to_problem_definition(ds.description)

    return dataset, problem_definition


def create_string_for_huggingface_dataset_card(
    description: dict,
    download_size_bytes: int,
    dataset_size_bytes: int,
    nb_samples: int,
    owner: str,
    license: str,
    zenodo_url: str = None,
    arxiv_paper_url: str = None,
    pretty_name: str = None,
    size_categories: list[str] = None,
    task_categories: list[str] = None,
    tags: list[str] = None,
    dataset_long_description: str = None,
    url_illustration: str = None,
) -> str:
    """Use this function for creating a dataset card, to upload together with the datase on the huggingface hub.

    Doing so ensure that load_dataset from the hub will populate the hf-dataset.description field, and be compatible for conversion to plaid.

    Without a dataset_card, the description field is lost.

    The parameters download_size_bytes and dataset_size_bytes can be determined after a
    dataset has been uploaded on huggingface:
    - manually by reading their values on the dataset page README.md,
    - automatically as shown in the example below

    See `the hugginface examples <https://github.com/PLAID-lib/plaid/blob/main/examples/bridges/huggingface_bridge_example.py>`__ for a concrete use.

    Args:
        description (dict): huggingface dataset description. Obtained from
        - description = hf_dataset.description
        - description = generate_huggingface_description(infos, problem_definition)
        download_size_bytes (int): the size of the dataset when downloaded from the hub
        dataset_size_bytes (int): the size of the dataset when loaded in RAM
        nb_samples (int): the number of samples in the dataset
        owner (str): the owner of the dataset, usually a username or organization name on huggingface
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
        problem_definition (ProblemDefinition): the problem definition generated from th huggingface dataset

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

This dataset contains a single huggingface split, named 'all_samples'.

The samples contains a single huggingface feature, named called "sample".

Samples are instances of [plaid.containers.sample.Sample](https://plaid-lib.readthedocs.io/en/latest/autoapi/plaid/containers/sample/index.html#plaid.containers.sample.Sample).
Mesh objects included in samples follow the [CGNS](https://cgns.github.io/) standard, and can be converted in
[Muscat.Containers.Mesh.Mesh](https://muscat.readthedocs.io/en/latest/_source/Muscat.Containers.Mesh.html#Muscat.Containers.Mesh.Mesh).


Example of commands:
```python
import pickle
from datasets import load_dataset
from plaid.containers.sample import Sample

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
