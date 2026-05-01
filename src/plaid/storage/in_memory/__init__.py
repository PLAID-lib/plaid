from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sequence, Union, overload
from pathlib import Path

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
        raise NotImplementedError("inMemoryBackend does not support init from disk")

    def download_from_hub(
        self,
        repo_id: str,
        local_dir: Union[str, Path],
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> str:
        raise NotImplementedError("InMemoryBackend download_from_hub not implemented")

    def init_datasetdict_streaming_from_hub(
        self,
        repo_id: str,
        split_ids: Optional[dict[str, Iterable[int]]] = None,
        features: Optional[list[str]] = None,
    ) -> dict[str, Any]:
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
        raise NotImplementedError("InMemoryBackend generate_to_disk not implemented")

    def push_local_to_hub(
        self, repo_id: str, local_dir: Union[str, Path], num_workers: int = 1
    ) -> None:
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
        samples: Union[Sample, Sequence[Sample]],
        id: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[int, list[int]]:
        if isinstance(samples, Sample):
            if id is None:
                id = len(self)
            elif not isinstance(id, int):
                raise TypeError("id must be a int if the sample is of type Sample")

            self.set_sample(sample=samples, id=id)
            return id
        elif isinstance(samples, Sequence):
            if id is None:
                id = list(range(len(self), len(self) + len(samples)))
            elif not isinstance(id, Sequence):
                raise TypeError(
                    "id must be a sequence if the sample is of type sequence"
                )
            else:
                if len(id) != len(np.unique(id)):
                    raise ValueError("the ids must be differents")

            if len(samples) != len(id):
                raise ValueError(
                    "The length of the list of samples to add and the list of IDs are different"
                )

            return [self.add_sample(samples=s, id=i) for i, s in zip(id, samples)]
        else:
            raise TypeError(
                f"sample must be a Sample of sequence[Samples], not : {type(samples)}"
            )

    @overload
    def set_sample(self, sample: Sample, id: Optional[int] = None) -> int: ...

    @overload
    def set_sample(
        self, sample: Sequence[Sample], id: Optional[Sequence[int]]
    ) -> list[int]: ...

    def set_sample(
        self,
        sample: Union[Sample, Sequence[Sample]],
        id: Optional[Union[int, Sequence[int]]] = None,
    ) -> Union[int, list[int]]:
        """Set the samples of the data set, overwriting the existing ones.

        Args:
            sample: A single sample or a sequence of samples to set.
            id: Optional single id or sequence of ids matching ``sample``.

        Raises:
            TypeError: If ``sample`` is not a :class:`Sample` or sequence of samples.
            TypeError: If ``id`` type does not match the ``sample`` kind.
            ValueError: If a provided integer id is negative.
        """
        if isinstance(sample, Sequence) and not isinstance(sample, Sample):
            if id is None:
                return [self.set_sample(s) for s in sample]
            if not isinstance(id, Sequence):  # pragma: no cover
                raise TypeError(
                    "id should be a sequence when sample is a sequence"
                )
            added_ids: list[int] = []
            for i, s in zip(id, sample):
                added_id = self.set_sample(sample=s, id=i)
                if not isinstance(added_id, int):  # pragma: no cover
                    raise TypeError("expected integer id when adding one sample")
                added_ids.append(added_id)
            return added_ids

        if not (isinstance(sample, Sample)):
            raise TypeError(f"sample should be of type Sample but is {type(sample)=}")

        if id is None:
            id = _find_first_missing(self._samples)
        elif not (isinstance(id, int)):
            raise TypeError(f"id should be of type {int.__class__} but {type(id)=}")

        if id < 0:
            raise ValueError(f"id should be positive (id>=0) but {id=}")

        self._samples[id] = sample

        return id

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
