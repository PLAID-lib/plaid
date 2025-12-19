# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Reader for hf dataset storage.

- If the environment variable `HF_ENDPOINT` is set, uses a private Hugging Face mirror.

    - Streaming is disabled.
    - The dataset is downloaded locally via `snapshot_download` and loaded from disk.

- If `HF_ENDPOINT` is not set, attempts to load from the public Hugging Face hub.

    - If the dataset is already cached locally, loads from disk.
    - Otherwise, loads from the hub, optionally using streaming mode.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import datasets
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download

from plaid.storage.common.reader import load_infos_from_disk

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------


def init_datasetdict_from_disk(path: Union[str, Path]) -> datasets.DatasetDict:
    """Initializes a DatasetDict from local disk files.

    Args:
        path (Union[str, Path]): Path to the directory containing the dataset files.

    Returns:
        datasets.DatasetDict: The loaded dataset dictionary.
    """
    file_ = Path(path) / "data" / "dataset_dict.json"
    if file_.is_file():
        # This is a dataset generated and save locally
        return load_from_disk(dataset_path=str(Path(path) / "data"))
    else:  # pragma: no cover
        # This is a dataset downloaded from the hub
        infos = load_infos_from_disk(path)
        split_names = list(infos["num_samples"].keys())
        base = Path(path) / "data"
        data_files = {sn: str(base / f"{sn}*.parquet") for sn in split_names}
        return load_dataset("parquet", data_files=data_files)


# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------


def download_datasetdict_from_hub(
    repo_id: str,
    local_dir: Union[str, Path],
    split_ids: Optional[dict[str, int]] = None,  # noqa: ARG001
    features: Optional[list[str]] = None,  # noqa: ARG001
    overwrite: bool = False,
) -> str:  # pragma: no cover (not tested in unit tests)
    """Downloads a dataset from Hugging Face Hub to local directory.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        local_dir (Union[str, Path]): Local directory to download to.
        split_ids (Optional[dict[str, int]]): Unused parameter for split selection.
        features (Optional[list[str]]): Unused parameter for feature selection.
        overwrite (bool): Whether to overwrite existing directory.

    Returns:
        str: Path to the downloaded dataset.
    """
    output_folder = Path(local_dir)

    if output_folder.is_dir():
        if overwrite:
            shutil.rmtree(output_folder)
            logger.warning(f"Existing {output_folder} directory has been reset.")
        elif any(output_folder.iterdir()):
            raise ValueError(
                f"directory {output_folder} already exists and is not empty. Set `overwrite` to True if needed."
            )

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/*"],
        local_dir=local_dir,
    )


def init_datasetdict_streaming_from_hub(
    repo_id: str,
    split_ids: Optional[dict[str, int]] = None,  # noqa: ARG001
    features: Optional[list[str]] = None,
) -> datasets.IterableDatasetDict:  # pragma: no cover
    """Initializes a streaming DatasetDict from Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub.
        split_ids (Optional[dict[str, int]]): Unused parameter for split selection.
        features (Optional[list[str]]): Optional list of features to load.

    Returns:
        datasets.IterableDatasetDict: The streaming dataset dictionary.
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if hf_endpoint:
        raise RuntimeError("Streaming mode not compatible with private mirror.")

    return load_dataset(repo_id, streaming=True, columns=features)
