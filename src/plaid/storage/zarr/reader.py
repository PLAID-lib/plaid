import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import fsspec
import yaml
import zarr
from huggingface_hub import hf_hub_download, snapshot_download

from .writer import flatten_path

logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------


def init_datasetdict_from_disk(
    path: Union[str, Path],
) -> dict[str, zarr.core.group.Group]:
    """Load a Hugging Face dataset or dataset dictionary from disk.

    This function wraps `datasets.load_from_disk` to accept either a string path or a
    `Path` object and returns the loaded dataset object.

    Args:
        path (Union[str, Path]): Path to the directory containing the saved dataset.
        *args:
            Positional arguments forwarded to
            [`datasets.load_from_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_from_disk).
        **kwargs:
            Keyword arguments forwarded to
            [`datasets.load_from_disk`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_from_disk).

    Returns:
        Union[datasets.Dataset, datasets.DatasetDict]: The loaded Hugging Face dataset
        object, which may be a single `Dataset` or a `DatasetDict` depending on what
        was saved on disk.
    """
    local_path = Path(path) / "data"
    split_names = [p.name for p in local_path.iterdir() if p.is_dir()]
    return {
        sn: zarr.open(zarr.storage.LocalStore(local_path / sn), mode="r")
        for sn in split_names
    }


# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------


def _zarr_patterns(
    repo_id,
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
):
    # include only selected sample ids
    if split_ids is not None:
        allow_patterns = []
        for split, ids in split_ids.items():
            allow_patterns.extend([f"data/{split}/zarr.json"])
            allow_patterns.extend([f"data/{split}/sample_{i:09d}/*" for i in ids])
    else:
        allow_patterns = ["data/*"]

    # ignore unwanted features
    ignore_patterns = []
    if features:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="variable_schema.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            variable_schema = yaml.safe_load(f)

        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="constant_schema.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            constant_schema = yaml.safe_load(f)

        all_features = list(variable_schema.keys()) + list(constant_schema.keys())
        ignored_features = [f for f in all_features if f not in features]

        ignore_patterns += [
            f"data/*/{flatten_path(feat)}/*" for feat in ignored_features
        ]

    return allow_patterns, ignore_patterns


def download_datasetdict_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
    overwrite: bool = False,
) -> None:  # pragma: no cover (not tested in unit tests)
    output_folder = Path(local_dir)

    if output_folder.is_dir():
        if overwrite:
            shutil.rmtree(local_dir)
            logger.warning(f"Existing {local_dir} directory has been reset.")
        elif any(local_dir.iterdir()):
            raise ValueError(
                f"directory {local_dir} already exists and is not empty. Set `overwrite` to True if needed."
            )

    allow_patterns, ignore_patterns = _zarr_patterns(repo_id, split_ids, features)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir=local_dir,
    )


class _LazyZarrArray:
    __slots__ = ("url", "fs", "_store")

    def __init__(self, url: str):
        self.url = url
        self.fs = fsspec.filesystem("https")
        self._store = None  # will be set lazily

    @property
    def store(self):
        if self._store is None:
            mapper = fsspec.get_mapper(self.url)
            self._store = zarr.open(mapper, mode="r")
        return self._store

    @property
    def ndim(self):
        return self.store.ndim

    @property
    def shape(self):
        return self.store.shape

    def __getitem__(self, key):
        return self.store[key]

    def __array__(self, dtype=None):
        arr = self.store[:]
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __call__(self):
        return self.store[:]

    def __repr__(self):
        return f"<_LazyZarrArray url={self.url} shape={self.store.shape} ndim={self.store.ndim}>"


def init_datasetdict_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
) -> dict[str, dict[int, dict[str, _LazyZarrArray]]]:
    """Lazily stream a Zarr dataset from a HF dataset repo.

    Returns:
        dataset[split][sample_id][feature] -> _LazyZarrArray
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if hf_endpoint:
        raise RuntimeError("Streaming mode not compatible with private mirror.")

    if features is not None:
        selected_features = features
    else:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="variable_schema.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            variable_schema = yaml.safe_load(f)
        selected_features = list(variable_schema.keys())

    if split_ids is not None:
        selected_ids = split_ids
    else:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="infos.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            infos = yaml.safe_load(f)
        selected_ids = {
            split: range(n_samples) for split, n_samples in infos["num_samples"].items()
        }

    dataset_dict: dict[str, dict[int, dict[str, _LazyZarrArray]]] = {}
    for split in selected_ids.keys():
        dataset_dict[split] = {}
        for sid in selected_ids[split]:
            dataset_dict[split][sid] = {}
            for feat in selected_features:
                flatten_feat = flatten_path(feat)
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{split}/sample_{sid:09d}/{flatten_feat}"
                dataset_dict[split][sid][feat] = _LazyZarrArray(url)

    return dataset_dict
