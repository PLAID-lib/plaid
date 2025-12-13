import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union, Iterator

import fsspec
import yaml
import zarr
from huggingface_hub import hf_hub_download, snapshot_download

from plaid.storage.zarr.bridge import unflatten_zarr_key
from plaid.storage.zarr.writer import flatten_path

from datasets import IterableDataset
from datasets.splits import NamedSplit

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Classes and functions
# ------------------------------------------------------

class ZarrDataset:
    def __init__(self, zarr_group: zarr.Group, **kwargs):
        self.zarr_group = zarr_group
        self._extra_fields =  dict(kwargs)

    def __iter__(self):
        for idx in self.ids:
            yield self[idx]

    def __getitem__(self, idx):
        zarr_sample = self.zarr_group[f"sample_{idx:09d}"]
        return {
            unflatten_zarr_key(path): zarr_sample[path] for path in zarr_sample.array_keys()
        }

    def __len__(self)->int:
        return len(self.zarr_group)

    def __getattr__(self, name):
        # fallback to extra fields
        if name in self._extra_fields:
            return self._extra_fields[name]
        # fallback to zarr_group attributes
        if hasattr(self.zarr_group, name):
            return getattr(self.zarr_group, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ('zarr_group', '_extra_fields'):
            super().__setattr__(name, value)
        else:
            self._extra_fields[name] = value

    def __repr__(self):
        return f"<ZarrDataset {repr(self.zarr_group)} | extra_fields={self._extra_fields}>"


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


def sample_generator(repo_id: str, split: str, ids: list[int], selected_features: list[str]) -> Iterator[dict]:

    base_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{split}/sample_"
    for idx in ids:
        sample = {}
        for feat in selected_features:
            flatten_feat = flatten_path(feat)
            mapper = fsspec.get_mapper(base_url + f"{idx:09d}/{flatten_feat}")
            sample[feat] = zarr.open(mapper, mode="r")

        yield sample


def create_zarr_iterable_dataset(repo_id, split, ids, selected_features):

    def wrapped_gen():
        yield from sample_generator(
            repo_id, split, ids, selected_features
        )

    return IterableDataset.from_generator(
        wrapped_gen,
        split=NamedSplit(split),
        features=None,
    )



# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------

def init_datasetdict_from_disk(
    path: Union[str, Path],
) -> dict[str, ZarrDataset]:
    local_path = Path(path) / "data"
    split_names = [p.name for p in local_path.iterdir() if p.is_dir()]
    return {
        sn: ZarrDataset(
            zarr.open(zarr.storage.LocalStore(local_path / sn), mode="r")
        )
        for sn in split_names
    }


# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------

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


def init_datasetdict_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,
) -> dict[str, IterableDataset]:
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

    return {split:create_zarr_iterable_dataset(repo_id, split, ids, selected_features) for split, ids in selected_ids.items()}