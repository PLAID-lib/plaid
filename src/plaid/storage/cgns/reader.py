import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union, Iterator

import fsspec
import numpy as np
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from plaid import Sample
from datasets import IterableDataset
from datasets.splits import NamedSplit

logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Classes and functions
# ------------------------------------------------------

class CGNSDataset:
    def __init__(self, path, **kwargs):

        self.path = path
        self._extra_fields = dict(kwargs)

        if Path(path).is_dir():
            sample_dirs = [
                p
                for p in path.iterdir()
                if p.is_dir() and p.name.startswith("sample_")
            ]
            sids = np.array([int(p.name.split("_")[1]) for p in sample_dirs], dtype=int)
            self._extra_fields["ids"] = np.sort(sids)
        else:
            raise ValueError("path mush be a local directory")

    def __iter__(self):
        for idx in self.ids:
            yield self[idx]

    def __getitem__(self, idx):
        assert idx in self.ids
        return Sample(path=self.path / f"sample_{idx:09d}")

    def __len__(self)->int:
        return len(self.ids)

    def __getattr__(self, name):
        # fallback to extra fields
        if name in self._extra_fields:
            return self._extra_fields[name]
        # fallback to zarr_group attributes
        if hasattr(self.path, name):
            return getattr(self.path, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ('path', '_extra_fields'):
            super().__setattr__(name, value)
        else:
            self._extra_fields[name] = value

    def __repr__(self):
        return f"<CGNSDataset {repr(self.path)} | extra_fields={self._extra_fields}>"


def sample_generator(repo_id: str, split: str, ids: list[int]) -> Iterator[Sample]:
    for idx in ids:
        with tempfile.TemporaryDirectory(prefix="plaid_sample_") as temp_folder:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=[f"data/{split}/sample_{idx:09d}/"],
                local_dir=temp_folder,
            )
            sample = Sample(
                path=Path(temp_folder)
                / "data"
                / f"{split}"
                / f"sample_{idx:09d}"
            )
        yield sample


def create_CGNS_iterable_dataset(repo_id: str,
                        split: str,
                        ids: list[int]) -> IterableDataset:

    return IterableDataset.from_generator(
        sample_generator,
        gen_kwargs={"repo_id": repo_id,
                    "split": split,
                    "ids": ids},
        split=NamedSplit(split),
        features=None,
    )


# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------

def init_datasetdict_from_disk(
    path: Union[str, Path],
) -> dict[str, CGNSDataset]:
    local_path = Path(path) / "data"
    split_names = [p.name for p in local_path.iterdir() if p.is_dir()]
    return {
        sn: CGNSDataset(local_path / sn)
        for sn in split_names
    }



# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------


def download_datasetdict_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None, # noqa: ARG001
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

    if split_ids is not None:
        allow_patterns = []
        for split, ids in split_ids.items():
            allow_patterns.extend([f"data/{split}/sample_{i:09d}/*" for i in ids])
    else:
        allow_patterns = ["data/*"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=local_dir,
    )


def init_datasetdict_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, int]] = None,
    features: Optional[list[str]] = None,  # noqa: ARG001
) -> dict[str, IterableDataset]:
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if hf_endpoint:
        raise RuntimeError("Streaming mode not compatible with private mirror.")

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

    return {split:create_CGNS_iterable_dataset(repo_id, split, ids) for split, ids in selected_ids.items()}