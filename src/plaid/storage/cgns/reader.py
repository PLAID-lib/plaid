import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Union

import fsspec
import numpy as np
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from plaid import Sample

logger = logging.getLogger(__name__)



# TODO: include _LazySampleStreaming in CGNSDataset

# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------


# class _LazySampleLocal:
#     __slots__ = ("_sample_path",)

#     def __init__(self, sample_path: Path):
#         self._sample_path = sample_path

#     def load(self):
#         return Sample(path=self._sample_path)

#     def __call__(self):
#         return self.load()

#     def __repr__(self):
#         return f"<LazySampleLocal path={self._sample_path}>"

class CGNSDataset:
    def __init__(self, path, **kwargs):

        super().__setattr__('path', path)
        super().__setattr__('_extra_fields', dict(kwargs))

        if Path(path).is_dir():
            sample_dirs = [
                p
                for p in path.iterdir()
                if p.is_dir() and p.name.startswith("sample_")
            ]
            sids = np.array([int(p.name.split("_")[1]) for p in sample_dirs], dtype=int)
            super().__setattr__('ids', np.sort(sids))
        else:
            raise ValueError("path mush be a local directory")

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


def init_datasetdict_from_disk(
    path: Union[str, Path],
) -> dict[str, CGNSDataset]:
    local_path = Path(path) / "data"
    split_names = [p.name for p in local_path.iterdir() if p.is_dir()]
    return {
        sn: CGNSDataset(local_path / sn)
        for sn in split_names
    }


    # path = Path(path) / "data"

    # split_ids = {}
    # for split in path.iterdir():
    #     if split.is_dir():
    #         sample_dirs = [
    #             p
    #             for p in split.iterdir()
    #             if p.is_dir() and p.name.startswith("sample_")
    #         ]
    #         sids = np.array([int(p.name.split("_")[1]) for p in sample_dirs], dtype=int)
    #         split_ids[split.name] = np.sort(sids)

    # dataset: dict[str, dict[int, _LazySampleLocal]] = {}

    # for split, ids in split_ids.items():
    #     split_path = path / split
    #     dataset[split] = {
    #         sid: _LazySampleLocal(split_path / f"sample_{sid:09d}") for sid in ids
    #     }

    # return dataset


# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------


class _LazySampleStreaming:
    __slots__ = ("repo_id", "split", "sid", "fs")

    def __init__(self, repo_id: str, split: str, sid: str):
        self.repo_id = repo_id
        self.split = split
        self.sid = sid
        self.fs = fsspec.filesystem("https")

    def load(self):
        with tempfile.TemporaryDirectory(prefix="plaid_sample_") as temp_folder:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                allow_patterns=[f"data/{self.split}/sample_{self.sid:09d}/"],
                local_dir=temp_folder,
            )
            sample = Sample(
                path=Path(temp_folder)
                / "data"
                / f"{self.split}"
                / f"sample_{self.sid:09d}"
            )
        return sample

    def __call__(self):
        return self.load()

    def __repr__(self):
        return f"<LazySampleStreaming repo={self.repo_id}, split={self.split}, id={self.sid}>"


def download_datasetdict_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, int]] = None,
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
    repo_id: str, split_ids: Optional[dict[str, int]] = None
) -> dict[str, dict[int, dict[str, _LazySampleStreaming]]]:
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

    dataset_dict: dict[str, dict[int, dict[str, _LazySampleStreaming]]] = {}
    for split in selected_ids.keys():
        dataset_dict[split] = {}
        for sid in selected_ids[split]:
            dataset_dict[split][sid] = _LazySampleStreaming(repo_id, split, sid)

    return dataset_dict
