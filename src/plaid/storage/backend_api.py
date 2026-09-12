"""Protocol definition for storage backend modules."""

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np
from datasets import IterableDataset

from .callbacks import SampleCallback

if TYPE_CHECKING:
    from ..containers.dataset import Dataset
    from ..containers.sample import Sample
    from ..infos import Infos


@runtime_checkable
class BackendModule(Protocol):
    """Protocol describing required methods for storage backend plugins."""

    name: str

    @staticmethod
    def init_from_disk(path: Union[str, Path]) -> Mapping[str, Any]:
        """Load a dataset dictionary from local storage.

        Args:
            path (Union[str, Path]): Path to the local dataset directory.

        Returns:
            Mapping[str, Any]: The loaded dataset dictionary.
        """
        ...

    @staticmethod
    def download_from_hub(
        repo_id: str,
        local_dir: Union[str, Path],
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,  # noqa: ARG001
        overwrite: bool = False,
    ) -> Path:
        """Download a dataset dictionary from a remote hub into a local folder.

        Args:
            repo_id (str): The repository ID on the remote hub.
            local_dir (Union[str, Path]): Local directory the dataset is
                downloaded into.
            split_ids (Optional[dict[str, Iterable[int]]]): Optional mapping of
                split names to the sample indices to download.
            features (Optional[list[str]]): Optional list of features to
                restrict the download to.
            overwrite (bool): Whether to overwrite an existing local directory.

        Returns:
            Path: The path to the downloaded dataset directory.
        """
        ...

    @staticmethod
    def init_datasetdict_streaming_from_hub(
        repo_id: str,
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,  # noqa: ARG001
    ) -> dict[str, IterableDataset]:
        """Initialize a streaming dataset dictionary from a remote hub.

        Args:
            repo_id (str): The repository ID on the remote hub.
            split_ids (Optional[dict[str, Iterable[int]]]): Optional mapping of
                split names to the sample indices to include in the stream.
            features (Optional[list[str]]): Optional list of features to load.

        Returns:
            dict[str, IterableDataset]: The streaming dataset dictionary,
                keyed by split name.
        """
        ...

    @staticmethod
    def generate_to_disk(
        output_folder: Union[str, Path],
        generators: dict[str, Callable[..., Generator["Sample", None, None]]],
        variable_schema: Optional[dict[str, dict]] = None,  # noqa: ARG001
        gen_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        num_proc: int = 1,
        verbose: bool = False,
        sample_callback: Optional[SampleCallback] = None,
        worker_initializer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Generate and save a dataset dictionary to local storage.

        ``sample_callback`` is called after each CGNS sample is written.

        Args:
            output_folder (Union[str, Path]): Directory the dataset is written
                into.
            generators (dict[str, Callable[..., Generator]]): Mapping of split
                names to sample generator callables.
            variable_schema (Optional[dict[str, dict]]): Optional schema
                describing the variable features of the dataset.
            gen_kwargs (Optional[dict[str, dict[str, Any]]]): Optional keyword
                arguments passed to each generator, keyed by split name.
            num_proc (int): Number of worker processes used to write samples.
            verbose (bool): Whether to display progress information.
            sample_callback (Optional[SampleCallback]): Callback invoked once
                per sample right after it is written to disk.
            worker_initializer (Optional[Callable[[], None]]): Optional
                callable run once in each worker process before generation.
        """
        ...

    @staticmethod
    def push_local_to_hub(
        repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
    ) -> None:
        """Push a local dataset dictionary to a remote hub repository.

        Args:
            repo_id (str): The repository ID on the remote hub.
            local_dir (Union[str, Path]): Local dataset directory to push.
            num_workers (int): Number of workers used for uploading.
        """
        ...

    @staticmethod
    def configure_dataset_card(
        repo_id: str,
        infos: "Infos",
        local_dir: Optional[Union[str, Path]] = None,
        viewer: bool = False,
        pretty_name: Optional[str] = None,
        dataset_long_description: Optional[str] = None,
        illustration_urls: Optional[list[str]] = None,
        arxiv_paper_urls: Optional[list[str]] = None,
    ) -> None:  # pragma: no cover
        """Configure metadata for a dataset card associated with a repository.

        Args:
            repo_id (str): The repository ID the dataset card is associated
                with.
            infos (Infos): Dataset metadata, including legal information such
                as the license.
            local_dir (Optional[Union[str, Path]]): Optional local directory
                holding the dataset being described.
            viewer (bool): Whether to enable the dataset viewer. Defaults to
                False.
            pretty_name (Optional[str]): A human-readable name for the dataset.
            dataset_long_description (Optional[str]): A detailed description of
                the dataset.
            illustration_urls (Optional[list[str]]): List of URLs to images
                illustrating the dataset.
            arxiv_paper_urls (Optional[list[str]]): List of arXiv URLs for
                papers related to the dataset.
        """
        ...

    @staticmethod
    def to_var_sample_dict(
        dataset: "Dataset",
        idx: int,
        features: Optional[list[str]] = None,
        indexers: Optional[dict[str, Any]] = None,
    ) -> dict[str, Optional[np.ndarray]]:
        """Convert a backend sample to PLAID variable-sample dictionary representation.

        Args:
            dataset (Dataset): The dataset to convert.
            idx (int): The sample index.
            features (Optional[list[str]]): Optional list of feature names to
                extract from the dataset.
            indexers (Optional[dict[str, Any]]): Optional mapping
                ``feature_path -> indexer`` used to select feature values along
                the last axis.

        Returns:
            dict[str, Optional[np.ndarray]]: The variable sample dictionary.
        """
        ...

    @staticmethod
    def sample_to_var_sample_dict(
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert a backend-native sample object to a variable-sample dictionary.

        Args:
            sample (dict[str, Any]): The backend-native sample dictionary.

        Returns:
            dict[str, Any]: The variable sample dictionary.
        """
        ...
