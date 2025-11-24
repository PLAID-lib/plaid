
from pathlib import Path
from typing import Union, Optional
import fsspec
import yaml
import shutil

import zarr
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download, hf_hub_download

from .writer import flatten_path

import logging
import numpy as np

from plaid.types.common import IndexType
logger = logging.getLogger(__name__)

#------------------------------------------------------
# Load from disk
#------------------------------------------------------

def load_dataset_from_disk(
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

def zarr_patterns(repo_id, split_ids:Optional[dict[str, IndexType]]=None, features: Optional[list[str]]=None):

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
            filename="key_mappings.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            key_mappings = yaml.safe_load(f)

        all_features = key_mappings["variable_features"] + key_mappings["constant_features"]
        ignored_features = [f for f in all_features if f not in features]

        ignore_patterns += [f"data/*/{flatten_path(feat)}/*" for feat in ignored_features]

    return allow_patterns, ignore_patterns


def download_dataset_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, IndexType]] = None,
    features: Optional[list[str]] = None,
    overwrite = False
)-> None:  # pragma: no cover (not tested in unit tests)

    local_dir = Path(local_dir)
    if local_dir.is_dir():
        if overwrite:
            shutil.rmtree(local_dir)
            logger.warning(f"Existing {local_dir} directory has been reset.")
        else:
            raise ValueError(
                f"directory {local_dir} already exists. Set `overwrite` to True if needed."
            )

    allow_patterns, ignore_patterns = zarr_patterns(repo_id, split_ids, features)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir=local_dir,
        local_files_only=False
    )


class LazyZarrArray:
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
        print(">>", self.url)
        arr = np.array(self.arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    # def __getitem__(self, key):
    #     return np.array(self.arr[key])


def stream_dataset_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, IndexType]] = None,
    features: Optional[list[str]] = None
) -> dict[str, dict[int, dict[str, LazyZarrArray]]]:
    """
    Lazily stream a Zarr dataset from a HF dataset repo.

    Returns:
        dataset[split][sample_id][feature] -> LazyZarrArray
    """
    if features is not None:
        selected_features = features
    else:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="key_mappings.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            key_mappings = yaml.safe_load(f)
        selected_features = key_mappings["variable_features"]

    if split_ids is not None:
        selected_ids = split_ids
    else:
        yaml_path = hf_hub_download(
            repo_id=repo_id,
            filename="split_n_samples.yaml",
            repo_type="dataset",
        )
        with open(yaml_path, "r", encoding="utf-8") as f:
            split_n_samples = yaml.safe_load(f)
        selected_ids = {split:range(n_samples) for split, n_samples in split_n_samples.items()}

    dataset: dict[str, dict[int, dict[str, LazyZarrArray]]] = {}
    for split in split_ids.keys():
        dataset[split] = {}
        for sid in selected_ids[split]:
            dataset[split][sid] = {}
            for feat in selected_features:
                flatten_feat = feat.replace("/", "__")
                url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{split}/sample_{sid:09d}/{flatten_feat}"
                dataset[split][sid][feat] = LazyZarrArray(url)

    return dataset

