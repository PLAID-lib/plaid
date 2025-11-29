import os

from pathlib import Path
from typing import Union, Optional
import fsspec
import yaml

import zarr
from huggingface_hub import snapshot_download, hf_hub_download

from .writer import flatten_path

import logging
import numpy as np

from plaid.types.common import IndexType
logger = logging.getLogger(__name__)

#------------------------------------------------------
# Load from disk
#------------------------------------------------------

def load_datasetdict(
    path: Union[str, Path]
)->dict[str, zarr.core.group.Group]:
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
    return {sn: zarr.open(zarr.storage.LocalStore(local_path / sn), mode="r") for sn in split_names}


#------------------------------------------------------
# Load from from hub
#------------------------------------------------------

def _zarr_patterns(repo_id, split_ids:Optional[dict[str, IndexType]]=None, features: Optional[list[str]]=None):

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

        ignore_patterns += [f"data/*/{flatten_path(feat)}/*" for feat in ignored_features]

    return allow_patterns, ignore_patterns


def download_datasetdict(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, IndexType]] = None,
    features: Optional[list[str]] = None,
)-> None:  # pragma: no cover (not tested in unit tests)

    allow_patterns, ignore_patterns = _zarr_patterns(repo_id, split_ids, features)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir=local_dir
    )


class _LazyZarrArray:
    def __init__(self, url: str):
        """
        Lazily loads a Zarr array from a HF dataset URL.

        url: HF dataset URL to the Zarr array (resolve/main)
        """
        self.url = url
        self.fs = fsspec.filesystem("https")
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            store = fsspec.get_mapper(f"{self.url}")
            self._arr = zarr.open(store, mode="r")
        return self._arr

    def __array__(self, dtype=None):
        arr = np.array(self.arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    # def __getitem__(self, key):
    #     return np.array(self.arr[key])


def init_streamed_datasetdict(
    repo_id: str,
    split_ids: Optional[dict[str, IndexType]] = None,
    features: Optional[list[str]] = None
) -> dict[str, dict[int, dict[str, _LazyZarrArray]]]:
    """
    Lazily stream a Zarr dataset from a HF dataset repo.

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
        selected_ids = {split:range(n_samples) for split, n_samples in infos["num_samples"].items()}

    dataset: dict[str, dict[int, dict[str, _LazyZarrArray]]] = {}
    for split in selected_ids.keys():
        dataset[split] = {}
        for sid in selected_ids[split]:
            dataset[split][sid] = {}
            for feat in selected_features:
                flatten_feat = feat.replace("/", "__")
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{split}/sample_{sid:09d}/{flatten_feat}"
                dataset[split][sid][feat] = _LazyZarrArray(url)

    return dataset

