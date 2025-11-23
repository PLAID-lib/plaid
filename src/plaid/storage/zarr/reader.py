
from pathlib import Path
from typing import Union, Optional
import fsspec

import zarr
from datasets import load_dataset, load_from_disk
from huggingface_hub import HfApi, snapshot_download

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


def discover_splits(repo_id):
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    splits = []
    for f in repo_files:
        if f.count("/") == 2 and f.endswith("zarr.json") and f.startswith("data/"):
            # data/<split>/zarr.json
            _, split, _ = f.split("/")
            splits.append(split)

    return sorted(set(splits))


# def zarr_partial_patterns(splits, ids=None, features=None):
#     patterns = []

#     for split in splits:
#         base = f"data/{split}"

#         # Always include split-level metadata
#         patterns.append(f"{base}/zarr.json")

#         # Case A — only ids
#         if ids is not None and not features:
#             for i in ids:
#                 sid = f"sample_{i:09d}"
#                 patterns += [
#                     f"{base}/{sid}/zarr.json",
#                     f"{base}/{sid}/*/zarr.json",
#                     f"{base}/{sid}/*",
#                 ]
#             continue

#         # Case B — only features (all samples)
#         if ids is None and features:
#             patterns.append(f"{base}/*/zarr.json")  # sample metadata
#             for feat in features:
#                 p = flatten_path(feat)
#                 patterns += [
#                     f"{base}/*/{p}/zarr.json",
#                     f"{base}/*/{p}/*",
#                 ]
#             continue

#         # Case C — both ids and features
#         if ids and features:
#             for i in ids:
#                 sid = f"sample_{i:09d}"
#                 patterns.append(f"{base}/{sid}/zarr.json")
#                 for feat in features:
#                     p = flatten_path(feat)
#                     patterns += [
#                         f"{base}/{sid}/{p}/zarr.json",
#                         f"{base}/{sid}/{p}/*",
#                     ]
#             continue

#         # Case D — no ids, no features → whole split
#         patterns.append(f"{base}/**")

#     return patterns




def zarr_partial_patterns(splits, ids=None, features=None):
    patterns = {}  # split -> list of patterns

    for split in splits:
        base = f"data/{split}"
        split_patterns = []

        split_patterns.append(f"{base}/zarr.json")

        if ids is not None and not features:
            for i in ids:
                sid = f"sample_{i:09d}"
                split_patterns += [
                    f"{base}/{sid}/zarr.json",
                    f"{base}/{sid}/*/zarr.json",
                    f"{base}/{sid}/*",
                ]

        elif ids is None and features:
            split_patterns.append(f"{base}/*/zarr.json")
            for feat in features:
                p = flatten_path(feat)
                split_patterns += [
                    f"{base}/*/{p}/zarr.json",
                    f"{base}/*/{p}/*",
                ]

        elif ids and features:
            for i in ids:
                sid = f"sample_{i:09d}"
                split_patterns.append(f"{base}/{sid}/zarr.json")
                for feat in features:
                    p = flatten_path(feat)
                    split_patterns += [
                        f"{base}/{sid}/{p}/zarr.json",
                        f"{base}/{sid}/{p}/*",
                    ]
        else:
            split_patterns.append(f"{base}/**")

        patterns[split] = split_patterns

    return patterns



def download_dataset_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    ids: Optional[IndexType] = None,
    features: Optional[list[str]] = None
)-> None:  # pragma: no cover (not tested in unit tests)

    splits = discover_splits(repo_id)
    split_patterns_dict = zarr_partial_patterns(splits, ids=ids, features=features)

    patterns = []
    for pat_list in split_patterns_dict.values():
        patterns.extend(pat_list)

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=local_dir,
        local_files_only=False
    )



class LazyZarrArray:
    def __init__(self, store_url: str, path_in_store: str):
        """
        Lazily loads a Zarr array from a HF dataset URL.

        store_url: HF dataset URL to the repo (resolve/main)
        path_in_store: path to the Zarr array (flattened __ version, root folder containing zarr.json)
        """
        self.store_url = store_url
        self.path_in_store = path_in_store
        self.fs = fsspec.filesystem("https")
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            store = fsspec.get_mapper(f"{self.store_url}/{self.path_in_store}")
            self._arr = zarr.open(store, mode="r")
        return self._arr

    def __array__(self, dtype=None):
        arr = np.array(self.arr)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __getitem__(self, key):
        return np.array(self.arr[key])


def stream_dataset_from_hub(
    repo_id: str,
    ids: Optional[list[int]] = None,
    features: Optional[list[str]] = None
) -> dict[str, dict[int, dict[str, LazyZarrArray]]]:
    """
    Lazily stream a Zarr dataset from a HF dataset repo.

    Returns:
        dataset[split][sample_id][feature] -> LazyZarrArray
    """
    from huggingface_hub import HfApi

    api = HfApi()
    repo_files = api.list_repo_files(repo_id, repo_type="dataset")

    # Only consider files under data/
    data_files = [f for f in repo_files if f.startswith("data/")]

    # Discover splits
    splits = sorted({f.split("/")[1] for f in data_files if len(f.split("/")) > 1})

    dataset: dict[str, dict[int, dict[str, LazyZarrArray]]] = {s: {} for s in splits}

    for f in data_files:
        parts = f.split("/")
        if len(parts) < 3:
            continue  # skip top-level zarr.json etc
        split, sample_dir = parts[1], parts[2]
        if not sample_dir.startswith("sample_"):
            continue
        sid = int(sample_dir.split("_")[1])
        if ids is not None and sid not in ids:
            continue

        if sid not in dataset[split]:
            dataset[split][sid] = {}

        # Skip split-level zarr.json
        if f.endswith("zarr.json") and len(parts) == 3:
            continue

        # Feature path is everything after sample_dir
        feat_path_flat = "/".join(parts[3:])

        # Remove trailing /c/... if present
        if "/c/" in feat_path_flat:
            feat_path_flat = feat_path_flat.split("/c/")[0]

        # Convert __ back to /
        feat_path_orig = feat_path_flat.replace("__", "/")

        # Optionally filter by features
        if features is not None and feat_path_orig not in features:
            continue

        dataset[split][sid][feat_path_orig] = LazyZarrArray(
            f"https://huggingface.co/datasets/{repo_id}/resolve/main",
            "/".join(parts[:3] + [feat_path_flat])
        )

    return dataset
