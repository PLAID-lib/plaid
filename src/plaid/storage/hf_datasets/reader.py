"""Reader for hf dataset storage.

- If the environment variable `HF_ENDPOINT` is set, uses a private Hugging Face mirror.

    - Streaming is disabled.
    - The dataset is downloaded locally via `snapshot_download` and loaded from disk.

- If `HF_ENDPOINT` is not set, attempts to load from the public Hugging Face hub.

    - If the dataset is already cached locally, loads from disk.
    - Otherwise, loads from the hub, optionally using streaming mode.
"""
import os
import shutil
from pathlib import Path
from typing import Union, Optional

import tempfile
import datasets
from datasets import load_dataset
from huggingface_hub import snapshot_download

#------------------------------------------------------
# Load from disk
#------------------------------------------------------

def load_datasetdict(
    path: Union[str, Path], **kwargs
) -> datasets.DatasetDict:
    return load_dataset(path = str(Path(path) / "data"), **kwargs)

#------------------------------------------------------
# Load from from hub
#------------------------------------------------------

def download_datasetdict(
    repo_id: str,
    local_dir: Union[str, Path],
)-> str:  # pragma: no cover (not tested in unit tests)

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/*"],
        local_dir=local_dir
    )


def init_streamed_datasetdict(
    repo_id: str,
    features: Optional[list[str]] = None,
    **kwargs
):
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if hf_endpoint:
        raise RuntimeError("Streaming mode not compatible with private mirror.")

    return load_dataset(repo_id, streaming=True, columns = features, **kwargs)