"""Protocol definition for storage backend modules."""

from pathlib import Path
from typing import Any, Protocol, Union, Iterable, Optional, Callable, Generator

import numpy as np

from datasets import IterableDataset

# from ..containers.dataset import Dataset
# from ..containers.sample import Sample
from ..types import IndexType


class BackendModule(Protocol):
    """Protocol describing required methods for storage backend plugins."""

    name: str

    def init_datasetdict_from_disk(self, path: Union[str, Path]) -> Any:
        """Load a dataset dictionary from local storage."""
        ...

    def download_datasetdict_from_hub(
        self,
        repo_id: str,
        local_dir: Union[str, Path],
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,  # noqa: ARG001
        overwrite: bool = False,
    ) -> str:
        """Download a dataset dictionary from a remote hub into a local folder."""
        ...

    def init_datasetdict_streaming_from_hub(
        self,
        repo_id: str,
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,  # noqa: ARG001
    ) -> dict[str, IterableDataset]:
        """Initialize a streaming dataset dictionary from a remote hub."""
        ...

    def generate_datasetdict_to_disk(
        self,
        output_folder: Union[str, Path],
        generators: dict[str, Callable[..., Generator["Sample", None, None]]],
        variable_schema: Optional[dict[str, dict]] = None,  # noqa: ARG001
        gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
        num_proc: int = 1,
        verbose: bool = False,
    ) -> None:
        """Generate and save a dataset dictionary to local storage."""
        ...

    def push_local_datasetdict_to_hub(
        self, repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
    ) -> None:
        """Push a local dataset dictionary to a remote hub repository."""
        ...

    def configure_dataset_card(
        self, repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
    ) -> None:  # pragma: no cover
        """Configure metadata for a dataset card associated with a repository."""
        ...

    def to_var_sample_dict(
        self,
        ds: "Dataset",
        i: int,
        features: Optional[list[str]] = None,
        enforce_shapes: bool = True,
    ) -> dict[str, Optional[np.ndarray]]:
        """"""
        ...

    def sample_to_var_sample_dict(
        self,
        hf_sample: dict[str, Any],
    ) -> dict[str, Any]:
        """"""
        ...
