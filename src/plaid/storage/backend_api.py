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
)

import numpy as np
from datasets import IterableDataset

from ..types import IndexType

if TYPE_CHECKING:
    from ..containers.dataset import Dataset
    from ..containers.sample import Sample


class BackendModule(Protocol):
    """Protocol describing required methods for storage backend plugins."""

    name: str

    @staticmethod
    def init_from_disk(path: Union[str, Path]) -> Mapping[str, Any]:
        """Load a dataset dictionary from local storage."""
        ...

    @staticmethod
    def download_from_hub(
        repo_id: str,
        local_dir: Union[str, Path],
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,  # noqa: ARG001
        overwrite: bool = False,
    ) -> str:
        """Download a dataset dictionary from a remote hub into a local folder."""
        ...

    @staticmethod
    def init_datasetdict_streaming_from_hub(
        repo_id: str,
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,  # noqa: ARG001
    ) -> dict[str, IterableDataset]:
        """Initialize a streaming dataset dictionary from a remote hub."""
        ...

    @staticmethod
    def generate_to_disk(
        output_folder: Union[str, Path],
        generators: dict[str, Callable[..., Generator["Sample", None, None]]],
        variable_schema: Optional[dict[str, dict]] = None,  # noqa: ARG001
        gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
        num_proc: int = 1,
        verbose: bool = False,
    ) -> None:
        """Generate and save a dataset dictionary to local storage."""
        ...

    @staticmethod
    def push_local_to_hub(
        repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
    ) -> None:
        """Push a local dataset dictionary to a remote hub repository."""
        ...

    @staticmethod
    def configure_dataset_card(
        repo_id: str,
        infos: dict[str, Any],
        local_dir: Optional[Union[str, Path]] = None,
        viewer: bool = False,
        pretty_name: Optional[str] = None,
        dataset_long_description: Optional[str] = None,
        illustration_urls: Optional[list[str]] = None,
        arxiv_paper_urls: Optional[list[str]] = None,
    ) -> None:  # pragma: no cover
        """Configure metadata for a dataset card associated with a repository."""
        ...

    @staticmethod
    def to_var_sample_dict(
        dataset: "Dataset",
        idx: int,
        features: Optional[list[str]] = None,
    ) -> dict[str, Optional[np.ndarray]]:
        """Convert a backend sample to PLAID variable-sample dictionary representation."""
        ...

    @staticmethod
    def sample_to_var_sample_dict(
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert a backend-native sample object to a variable-sample dictionary."""
        ...
