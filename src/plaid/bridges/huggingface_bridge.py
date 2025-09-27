"""Hugging Face bridge for PLAID datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
import io
import pickle
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional

import yaml
from tqdm import tqdm

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import logging
import os
from typing import Union

import datasets
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from pydantic import ValidationError

from plaid import Dataset, ProblemDefinition, Sample
from plaid.containers.features import SampleFeatures
from plaid.utils.deprecation import deprecated

logger = logging.getLogger(__name__)

"""
Convention with hf (Hugging Face) datasets:
- samples contains a single Hugging Face feature, named called "sample".
- Samples are instances of :ref:`Sample`.
- Mesh objects included in samples follow the CGNS standard, and can be converted in Muscat.Containers.Mesh.Mesh.
- problem_definition info is stored in hf-datasets "description" parameter
"""


# ------------------------------------------------------------------------------
def load_hf_dataset_from_hub(
    repo_id: str, streaming: bool = False, *args, **kwargs
) -> Union[
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict,
]:  # pragma: no cover (to prevent testing from downloading, this is run by examples)
    """Loads a Hugging Face dataset from the public hub, a private mirror, or local cache, with automatic handling of streaming and download modes.

    Behavior:

    - If the environment variable `HF_ENDPOINT` is set, uses a private Hugging Face mirror.

      - Streaming is disabled.
      - The dataset is downloaded locally via `snapshot_download` and loaded from disk.

    - If `HF_ENDPOINT` is not set, attempts to load from the public Hugging Face hub.

      - If the dataset is already cached locally, loads from disk.
      - Otherwise, loads from the hub, optionally using streaming mode.

    Args:
        repo_id (str): The Hugging Face dataset repository ID (e.g., 'username/dataset').
        streaming (bool, optional): If True, attempts to stream the dataset (only supported on the public hub).
        *args: Additional positional arguments passed to `datasets.load_dataset` or `datasets.load_from_disk`.
        **kwargs: Additional keyword arguments passed to `datasets.load_dataset` or `datasets.load_from_disk`.

    Returns:
        Union[datasets.Dataset, datasets.DatasetDict]: The loaded Hugging Face dataset object.

    Raises:
        Exception: Propagates any exceptions raised by `datasets.load_dataset`, `datasets.load_from_disk`, or `huggingface_hub.snapshot_download` if loading fails.

    Notes:
        - Streaming mode is not supported when using a private mirror.
        - If the dataset is found in the local cache, loads from disk instead of streaming.
        - To use behind a proxy or with a private mirror, you may need to set:
            - HF_ENDPOINT to your private mirror address
            - CURL_CA_BUNDLE to your trusted CA certificates
            - HF_HOME to a shared cache directory if needed
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()

    # Helper to check if dataset repo is already cached
    def _get_cached_path(repo_id_):
        try:
            return snapshot_download(
                repo_id=repo_id_, repo_type="dataset", local_files_only=True
            )
        except FileNotFoundError:
            return None

    # Private mirror case
    if hf_endpoint:
        if streaming:
            logger.warning(
                "Streaming mode not compatible with private mirror. Falling back to download mode."
            )
        local_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
        return load_dataset(local_path, *args, **kwargs)

    # Public case
    local_path = _get_cached_path(repo_id)
    if local_path is not None and streaming is True:
        # Even though streaming mode: rely on local files if already downloaded
        logger.info("Dataset found in cache. Loading from disk instead of streaming.")
        return load_dataset(local_path, *args, **kwargs)

    return load_dataset(repo_id, streaming=streaming, *args, **kwargs)


def load_hf_infos_from_hub(
    repo_id: str,
) -> dict[
    str, dict[str, str]
]:  # pragma: no cover (to prevent testing from downloading, this is run by examples)
    """Load dataset infos from the Hugging Face Hub.

    Downloads the infos.yaml file from the specified repository and parses it as a dictionary.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing dataset infos.
    """
    # Download infos.yaml
    yaml_path = hf_hub_download(
        repo_id=repo_id, filename="infos.yaml", repo_type="dataset"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        infos = yaml.safe_load(f)

    return infos


def load_hf_problem_definition_from_hub(
    repo_id: str, name: str
) -> (
    ProblemDefinition
):  # pragma: no cover (to prevent testing from downloading, this is run by examples)
    """Load a ProblemDefinition from the Hugging Face Hub.

    Downloads the problem infos YAML and split JSON files from the specified repository and location,
    then initializes a ProblemDefinition object with this information.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        name (str): The name of the problem_definition stored in the repo.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    # Download split.json
    # json_path = hf_hub_download(
    #     repo_id=repo_id,
    #     filename=f"problem_definitions/{name}/split.json",
    #     repo_type="dataset",
    # )
    # with open(json_path, "r", encoding="utf-8") as f:
    #     json_data = json.load(f)

    # Download problem_infos.yaml
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"problem_definitions/{name}/problem_infos.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    prob_def = ProblemDefinition()
    prob_def._initialize_from_problem_infos_dict(yaml_data)
    # prob_def.set_split(json_data)

    return prob_def


def push_dataset_dict_to_hub(
    repo_id: str, hf_dataset_dict: datasets.DatasetDict
) -> None:  # pragma: no cover (push not tested)
    """Push a Hugging Face dataset to the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        hf_dataset_dict (datasets.Dataset): The Hugging Face dataset to push.
    """
    hf_dataset_dict.push_to_hub(repo_id)


def push_dataset_infos_to_hub(
    repo_id: str, infos: dict[str, dict[str, str]]
) -> None:  # pragma: no cover (push not tested)
    """Upload dataset infos to the Hugging Face Hub.

    Serializes the infos dictionary to YAML and uploads it to the specified repository as data/infos.yaml.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset infos to upload.

    Raises:
        ValueError: If the infos dictionary is empty.
    """
    if len(infos) > 0:
        api = HfApi()
        yaml_str = yaml.dump(infos)
        yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))
        api.upload_file(
            path_or_fileobj=yaml_buffer,
            path_in_repo="infos.yaml",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload infos.yaml",
        )
    else:
        raise ValueError("'infos' must not be empty")


def push_problem_definition_to_hub(
    repo_id: str, name: str, pb_def: ProblemDefinition
) -> None:  # pragma: no cover (push not tested)
    """Upload a ProblemDefinition and its split information to the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        name (str): The name of the problem_definition to store in the repo.
        pb_def (ProblemDefinition): The problem definition to upload.
    """
    api = HfApi()
    data = pb_def._generate_problem_infos_dict()
    if data is not None:
        yaml_str = yaml.dump(data)
        yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo=f"problem_definitions/{name}/problem_infos.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload problem_definitions/{name}/problem_infos.yaml",
    )

    # data = pb_def.get_split()
    # json_str = json.dumps(data)
    # json_buffer = io.BytesIO(json_str.encode("utf-8"))

    # api.upload_file(
    #     path_or_fileobj=json_buffer,
    #     path_in_repo=f"problem_definitions/{name}/split.json",
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     commit_message=f"Upload problem_definitions/{name}/split.json",
    # )


# ------------------------------------------------------------------------------


def load_dataset_dict_from_to_disk(path: Union[str, Path]) -> datasets.DatasetDict:
    """Load a Hugging Face DatasetDict from disk.

    Args:
        path (Union[str, Path]): The directory path from which to load the dataset dict.

    Returns:
        datasets.DatasetDict: The loaded Hugging Face DatasetDict.
    """
    return load_from_disk(str(path))


def load_dataset_infos_from_disk(path: Union[str, Path]) -> dict[str, dict[str, str]]:
    """Load dataset infos from a YAML file on disk.

    Args:
        path (Union[str, Path]): The directory path containing the infos file.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing dataset infos.
    """
    infos_fname = Path(path) / "infos.yaml"
    with infos_fname.open("r") as file:
        infos = yaml.safe_load(file)
    return infos


def load_problem_definition_from_disk(
    path: Union[str, Path], name: Union[str, Path]
) -> ProblemDefinition:
    """Load a ProblemDefinition and its split information from disk.

    Args:
        path (Union[str, Path]): The root directory path for loading.
        name (str): The name of the problem_definition stored in the disk directory.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    pb_def = ProblemDefinition()
    pb_def._load_from_dir_(Path(path) / Path("problem_definitions") / Path(name))
    pb_def.set_split({})
    return pb_def


def save_dataset_dict_to_disk(
    path: Union[str, Path], hf_dataset_dict: datasets.DatasetDict
) -> None:
    """Save a Hugging Face DatasetDict to disk.

    Args:
        path (Union[str, Path]): The directory path where the dataset dict will be saved.
        hf_dataset_dict (datasets.DatasetDict): The Hugging Face DatasetDict to save.
    """
    hf_dataset_dict.save_to_disk(str(path))


def save_dataset_infos_to_disk(
    path: Union[str, Path], infos: dict[str, dict[str, str]]
) -> None:
    """Save dataset infos as a YAML file to disk.

    Args:
        path (Union[str, Path]): The directory path where the infos file will be saved.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset infos.
    """
    infos_fname = Path(path) / "infos.yaml"
    infos_fname.parent.mkdir(parents=True, exist_ok=True)
    with open(infos_fname, "w") as file:
        yaml.dump(infos, file, default_flow_style=False, sort_keys=False)


def save_problem_definition_to_disk(
    path: Union[str, Path], name: Union[str, Path], pb_def: ProblemDefinition
) -> None:
    """Save a ProblemDefinition and its split information to disk.

    Args:
        path (Union[str, Path]): The root directory path for saving.
        name (str): The name of the problem_definition to store in the disk directory.
        pb_def (ProblemDefinition): The problem definition to save.
    """
    pb_def.set_split({})
    pb_def._save_to_dir_(Path(path) / Path("problem_definitions") / Path(name))


# ------------------------------------------------------------------------------


def to_plaid_sample(hf_sample: dict[str, bytes]) -> Sample:
    """Convert a Hugging Face dataset sample to a plaid :class:`Sample <plaid.containers.sample.Sample>`.

    If the sample is not valid, it tries to build it from its components.
    If it still fails because of a missing key, it raises a KeyError.
    """
    pickled_hf_sample = pickle.loads(hf_sample["sample"])

    try:
        # Try to validate the sample
        return Sample.model_validate(pickled_hf_sample)

    except ValidationError:
        features = SampleFeatures(
            data=pickled_hf_sample.get("meshes"),
        )

        sample = Sample(
            path=pickled_hf_sample.get("path"),
            features=features,
        )

        scalars = pickled_hf_sample.get("scalars")
        if scalars:
            for sn, val in scalars.items():
                sample.add_scalar(sn, val)

        return Sample.model_validate(sample)


def plaid_dataset_to_huggingface(
    dataset: Dataset,
    problem_definition: Optional[ProblemDefinition] = None,
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
    if problem_definition is None and split == "all_samples":
        ids = dataset.get_sample_ids()
    else:
        if split == "all_samples":
            ids = range(len(dataset))
        else:
            assert problem_definition is not None, (
                "if split is not 'all_samples', problem_definition must be set."
            )
            ids = problem_definition.get_split(split)

    def generator():
        for sample in dataset[ids]:
            yield {
                "sample": pickle.dumps(sample.model_dump()),
            }

    return plaid_generator_to_huggingface(
        generator=generator,
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
    split: str = "all_samples",
    processes_number: int = 1,
) -> datasets.Dataset:
    """Use this function for creating a Hugging Face dataset from a sample generator function.

    This function can be used when the plaid dataset cannot be loaded in RAM all at once due to its size.
    The generator enables loading samples one by one.

    Args:
        generator (Callable): a function yielding a dict {"sample" : sample}, where sample is of type 'bytes'
        infos (dict):  the info is used to generate the description of the Hugging Face dataset.
        split (str): The name of the split. Default: "all_samples".
        processes_number (int): The number of processes used to generate the Hugging Face dataset. Default: 1.

    Returns:
        datasets.Dataset: dataset in Hugging Face format

    Example:
        .. code-block:: python

            dataset = plaid_generator_to_huggingface(generator, infos, split)
    """
    ds: datasets.Dataset = datasets.Dataset.from_generator(  # pyright: ignore[reportAssignmentType]
        generator,
        features=datasets.Features({"sample": datasets.Value("binary")}),
        num_proc=processes_number,
        writer_batch_size=1,
        split=datasets.splits.NamedSplit(split),
    )

    return ds


def plaid_generator_to_huggingface_datasetdict(
    generator: Callable,
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

            hf_dataset_dict = plaid_generator_to_huggingface_datasetdict(generator, infos, problem_definition, main_splits)
            push_dataset_dict_to_hub("chanel/dataset", hf_dataset_dict)
            hf_dataset_dict.save_to_disk("path/to/dir")
    """
    _dict = {}
    for _, split in enumerate(main_splits):
        ds = plaid_generator_to_huggingface(
            generator,
            split=split,
            processes_number=processes_number,
        )
        _dict[split] = ds

    return datasets.DatasetDict(_dict)


def huggingface_dataset_to_plaid(
    ds: datasets.Dataset,
    ids: Optional[list[int]] = None,
    processes_number: int = 1,
    large_dataset: bool = False,
    verbose: bool = True,
) -> Dataset:
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
    from plaid.bridges.huggingface_helpers import (
        _HFShardToPlaidSampleConverter,
        _HFToPlaidSampleConverter,
    )

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
                        pool.imap(converter, range(len(converter.hf_ds))),
                        total=len(converter.hf_ds),
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

    return dataset


@deprecated("will be removed (no alternative)", version="0.1.9", removal="0.2.0")
def huggingface_description_to_problem_definition(
    description: dict,
) -> ProblemDefinition:
    """Converts a Hugging Face dataset description to a plaid problem definition.

    Args:
        description (dict): the description field of a Hugging Face dataset, containing the problem definition

    Returns:
        problem_definition (ProblemDefinition): the plaid problem definition initialized from the Hugging Face dataset description
    """
    description = {} if description == "" else description
    problem_definition = ProblemDefinition()
    for func, key in [
        (problem_definition.set_task, "task"),
        (problem_definition.set_split, "split"),
        (problem_definition.add_input_scalars_names, "in_scalars_names"),
        (problem_definition.add_output_scalars_names, "out_scalars_names"),
        (problem_definition.add_input_fields_names, "in_fields_names"),
        (problem_definition.add_output_fields_names, "out_fields_names"),
        (problem_definition.add_input_meshes_names, "in_meshes_names"),
        (problem_definition.add_output_meshes_names, "out_meshes_names"),
    ]:
        try:
            func(description[key])
        except KeyError:
            pass

    return problem_definition


@deprecated("will be removed (no alternative)", version="0.1.9", removal="0.2.0")
def huggingface_description_to_infos(
    description: dict,
) -> dict[str, dict[str, str]]:
    """Convert a Hugging Face dataset description dictionary to a PLAID infos dictionary.

    Extracts the "legal" and "data_production" sections from the Hugging Face description
    and returns them in a format compatible with PLAID dataset infos.

    Args:
        description (dict): The Hugging Face dataset description dictionary.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing "legal" and "data_production" infos if present.
    """
    infos = {}
    if "legal" in description:
        infos["legal"] = description["legal"]
    if "data_production" in description:
        infos["data_production"] = description["data_production"]
    return infos


@deprecated("will be removed (no alternative)", version="0.1.9", removal="0.2.0")
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
