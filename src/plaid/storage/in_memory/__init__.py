from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Union,
    overload,
)

import numpy as np

from ...containers.sample import Sample
from ...types import IndexType


def _find_first_missing(d: Iterable[int]) -> int:
    key = 0  # Or 0, depending on your starting preference
    while key in d:
        key += 1
    return key


class InMemoryBackend:
    name = "in_memory"

    @staticmethod
    def init_from_disk(path: Union[str, Path]) -> Any:
        """Raise because the in-memory backend cannot be initialized from disk."""
        raise NotImplementedError("inMemoryBackend does not support init from disk")

    def download_from_hub(
        self,
        repo_id: str,
        local_dir: Union[str, Path],
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> str:
        """Raise because hub download is not implemented for the in-memory backend."""
        raise NotImplementedError("InMemoryBackend download_from_hub not implemented")

    def init_datasetdict_streaming_from_hub(
        self,
        repo_id: str,
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Raise because streaming from hub is not implemented for this backend."""
        raise NotImplementedError(
            "InMemoryBackend init_datasetdict_streaming_from_hub not implemented"
        )

    def generate_to_disk(
        self,
        output_folder: Union[str, Path],
        generators: dict[str, Callable[..., Generator[Sample, None, None]]],
        variable_schema: Optional[dict[str, dict]] = None,
        gen_kwargs: Optional[dict[str, dict[str, list[IndexType]]]] = None,
        num_proc: int = 1,
        verbose: bool = False,
    ) -> None:
        """Raise because writing to disk is not implemented for the in-memory backend."""
        raise NotImplementedError("InMemoryBackend generate_to_disk not implemented")

    def push_local_to_hub(
        self, repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
    ) -> None:
        """Raise because pushing to hub is not implemented for this backend."""
        raise NotImplementedError("InMemoryBackend push_local_to_hub not implemented")

    def configure_dataset_card(
        self,
        repo_id: str,
        infos: dict,
        local_dir: Optional[Union[str, Path]] = None,
        viewer: bool = False,
        pretty_name: Optional[str] = None,
        dataset_long_description: Optional[str] = None,
        illustration_urls: Optional[list[str]] = None,
        arxiv_paper_urls: Optional[list[str]] = None,
    ) -> None:
        """Raise because dataset-card configuration is not implemented for this backend."""
        raise NotImplementedError(
            "InMemoryBackend configure_dataset_card not implemented"
        )

    def __init__(self) -> None:
        self._samples: Dict[int, Sample] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, key: Union[int, slice, Sequence[int]]
    ) -> Union[Sample, list[Sample]]:
        if isinstance(key, (slice, Sequence)):
            return [
                self._samples[k]
                for k in (
                    range(*key.indices(len(self))) if isinstance(key, slice) else key
                )
            ]
        return self._samples[key]

    def add_sample(
        self,
        sample: Union[Sample, Sequence[Sample]],
        sample_id: Optional[Union[int, Sequence[int]]] = None,
        *,
        id: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[int, list[int]]:
        """Add one sample or a sequence of samples to the in-memory backend.

        Args:
            sample: One :class:`Sample` or a sequence of samples.
            sample_id: Optional id(s) associated with ``sample``.
            id: Alias of ``sample_id`` for backward compatibility.

        Returns:
            Added sample id or list of added ids.
        """
        if sample_id is None:
            sample_id = id

        if isinstance(sample, Sample):
            if sample_id is None:
                sample_id = len(self)
            elif not isinstance(sample_id, int):
                raise TypeError("sample_id must be an int when samples is a Sample")

            self.set_sample(sample=sample, sample_id=sample_id)
            return sample_id
        elif isinstance(sample, Sequence):
            if sample_id is None:
                sample_id = list(range(len(self), len(self) + len(sample)))
            elif not isinstance(sample_id, Sequence):
                raise TypeError(
                    "sample_id must be a sequence when samples is a sequence"
                )
            else:
                if len(sample_id) != len(np.unique(sample_id)):
                    raise ValueError("sample_ids must be unique")

            if len(sample) != len(sample_id):
                raise ValueError(
                    "The length of the list of samples to add and the list of IDs are different"
                )

            return [
                self.add_sample(sample=s, sample_id=i)
                for i, s in zip(sample_id, sample)
            ]
        else:
            raise TypeError(
                f"sample must be a Sample of sequence[Samples], not : {type(sample)}"
            )

    @overload
    def set_sample(self, sample: Sample, sample_id: Optional[int] = None) -> int: ...

    @overload
    def set_sample(
        self, sample: Sequence[Sample], sample_id: Optional[Sequence[int]]
    ) -> list[int]: ...

    def set_sample(
        self,
        sample: Union[Sample, Sequence[Sample]],
        sample_id: Optional[Union[int, Sequence[int]]] = None,
        *,
        id: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[int, list[int]]:
        """Set the samples of the data set, overwriting the existing ones.

        Args:
            sample: A single sample or a sequence of samples to set.
            sample_id: Optional single id or sequence of ids matching ``sample``.

        Raises:
            TypeError: If ``sample`` is not a :class:`Sample` or sequence of samples.
            TypeError: If ``sample_id`` type does not match the ``sample`` kind.
            ValueError: If a provided integer sample_id is negative.
        """
        if sample_id is None:
            sample_id = id

        if isinstance(sample, Sequence) and not isinstance(sample, Sample):
            if sample_id is None:
                return [self.set_sample(s) for s in sample]
            if not isinstance(sample_id, Sequence):  # pragma: no cover
                raise TypeError(
                    "sample_id should be a sequence when sample is a sequence"
                )
            added_ids: list[int] = []
            for i, s in zip(sample_id, sample):
                added_id = self.set_sample(sample=s, sample_id=i)
                if not isinstance(added_id, int):  # pragma: no cover
                    raise TypeError("expected integer id when adding one sample")
                added_ids.append(added_id)
            return added_ids

        if not (isinstance(sample, Sample)):
            raise TypeError(f"sample should be of type Sample but is {type(sample)=}")

        if sample_id is None:
            sample_id = _find_first_missing(self._samples)
        elif not (isinstance(sample_id, int)):
            raise TypeError(
                f"sample_id should be of type {int.__class__} but {type(sample_id)=}"
            )

        if sample_id < 0:
            raise ValueError(f"sample_id should be positive (sample_id>=0) but {sample_id=}")

        self._samples[sample_id] = sample

        return sample_id

    def merge_dataset(self, dataset: Any) -> Optional[list[int]]:
        """Merges samples of another dataset into this one.

        Args:
            dataset (Dataset): The data set to be merged into this one (self).
            in_place (bool, option): If True, modifies the current dataset in place.

        Returns:
            Optional[list[int]]: ids of added :class:`Samples <plaid.containers.sample.Sample>`
            from input :class:`Dataset <plaid.containers.dataset.Dataset>`. Returns
            ``None`` when ``dataset`` is ``None``.

        Raises:
            ValueError: If the provided dataset value is not an instance of Dataset
        """
        if dataset is None:
            return None

        added_ids: list[int] = []
        for i in range(len(dataset)):
            added_id = self.add_sample(dataset[i])
            if not isinstance(added_id, int):  # pragma: no cover
                raise TypeError("expected integer id when merging dataset samples")
            added_ids.append(added_id)
        return added_ids
