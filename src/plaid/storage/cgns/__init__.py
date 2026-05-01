"""Package for CGNS storage."""
from .reader import (
    download_datasetdict_from_hub,
    init_datasetdict_from_disk,
    init_datasetdict_streaming_from_hub,
)
from .writer import (
    configure_dataset_card,
    generate_datasetdict_to_disk,
    push_local_datasetdict_to_hub,
)

from typing import Optional, Union, Mapping, Any
from pathlib import Path
from collections.abc import Iterable

from datasets import IterableDataset


class CgnsBackend:
    name = "cgns"

    @staticmethod
    def init_from_disk(path: Union[str, Path]) -> Mapping[str, Any]:
        return init_datasetdict_from_disk(path=path)

    @staticmethod
    def download_from_hub(
        repo_id: str,
        local_dir: Union[str, Path],
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> str:
        return download_datasetdict_from_hub(
            repo_id=repo_id,
            local_dir=local_dir,
            split_ids=split_ids,
            features=features,
            overwrite=overwrite,
        )

    @staticmethod
    def init_datasetdict_streaming_from_hub(
        repo_id: str,
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,
    ) -> dict[str, IterableDataset]:
        return init_datasetdict_streaming_from_hub(
            repo_id=repo_id, split_ids=split_ids, features=features
        )

    @staticmethod
    def generate_to_disk(
        output_folder: Union[str, Path],
        generators: dict,
        variable_schema: Optional[dict[str, dict]] = None,
        gen_kwargs: Optional[dict[str, dict[str, list]]] = None,
        num_proc: int = 1,
        verbose: bool = False,
    ) -> None:
        return generate_datasetdict_to_disk(
            output_folder=output_folder,
            generators=generators,
            variable_schema=variable_schema,
            gen_kwargs=gen_kwargs,
            num_proc=num_proc,
            verbose=verbose,
        )

    @staticmethod
    def push_local_to_hub(
        repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
    ) -> None:
        return push_local_datasetdict_to_hub(
            repo_id=repo_id, local_dir=local_dir, num_workers=num_workers
        )

    @staticmethod
    def configure_dataset_card(
        repo_id: str,
        infos: dict,
        local_dir: Optional[Union[str, Path]] = None,
        viewer: bool = False,
        pretty_name: Optional[str] = None,
        dataset_long_description: Optional[str] = None,
        illustration_urls: Optional[list[str]] = None,
        arxiv_paper_urls: Optional[list[str]] = None,
    ) -> None:
        if local_dir is None:
            raise ValueError("local_dir must be provided for cgns backend")
        return configure_dataset_card(
            repo_id=repo_id,
            infos=infos,
            local_dir=local_dir,
            viewer=viewer,
            pretty_name=pretty_name,
            dataset_long_description=dataset_long_description,
            illustration_urls=illustration_urls,
            arxiv_paper_urls=arxiv_paper_urls,
        )

    @staticmethod
    def to_var_sample_dict(
        dataset: object, idx: int, features: Optional[list[str]] = None
    ) -> dict:
        raise ValueError(f"to_dict not available for 'cgns' backend")

    @staticmethod
    def sample_to_var_sample_dict(sample: dict) -> dict:
        raise ValueError(f"sample_to_var_sample_dict not available for 'cgns' backend")


__all__ = [
    "configure_dataset_card",
    "download_datasetdict_from_hub",
    "generate_datasetdict_to_disk",
    "init_datasetdict_from_disk",
    "init_datasetdict_streaming_from_hub",
    "push_local_datasetdict_to_hub",
]
