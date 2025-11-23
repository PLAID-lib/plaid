
import os
from pathlib import Path
from typing import Union

import datasets
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download

import logging
logger = logging.getLogger(__name__)

#------------------------------------------------------
# Load from disk
#------------------------------------------------------

def load_dataset_from_disk(
    path: Union[str, Path], *args, **kwargs
) -> Union[datasets.Dataset, datasets.DatasetDict]:
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
    return load_from_disk(str(path), *args, **kwargs)


#------------------------------------------------------
# Load from from hub
#------------------------------------------------------


def load_dataset_from_hub(
    repo_id: str, streaming: bool = False, *args, **kwargs
) -> Union[
    datasets.Dataset,
    datasets.DatasetDict,
    datasets.IterableDataset,
    datasets.IterableDatasetDict,
]:  # pragma: no cover (not tested in unit tests)
    """Loads a Hugging Face dataset from the public hub, a private mirror, or local cache, with automatic handling of streaming and download modes.

    Behavior:

    - If the environment variable `HF_ENDPOINT` is set, uses a private Hugging Face mirror.

      - Streaming is disabled.
      - The dataset is downloaded locally via `snapshot_download` and loaded from disk.

    - If `HF_ENDPOINT` is not set, attempts to load from the public Hugging Face hub.

      - If the dataset is already cached locally, loads from disk.
      - Otherwise, loads from the hub, optionally using streaming mode.

    Args:
        repo_id (str): The Hugging Face dataset repository ID (e.g., 'username/dataset').
        streaming (bool, optional): If True, attempts to stream the dataset (only supported on the public hub).
        *args:
            Positional arguments forwarded to
            [`datasets.load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset).
        **kwargs:
            Keyword arguments forwarded to
            [`datasets.load_dataset`](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset).

    Returns:
        Union[datasets.Dataset, datasets.DatasetDict]: The loaded Hugging Face dataset object.

    Raises:
        Exception: Propagates any exceptions raised by `datasets.load_dataset`, `datasets.load_from_disk`, or `huggingface_hub.snapshot_download` if loading fails.

    Note:
        - Streaming mode is not supported when using a private mirror.
        - If the dataset is found in the local cache, loads from disk instead of streaming.
        - To use behind a proxy or with a private mirror, you may need to set:
            - HF_ENDPOINT to your private mirror address
            - CURL_CA_BUNDLE to your trusted CA certificates
            - HF_HOME to a shared cache directory if needed
    """
    hf_endpoint = os.getenv("HF_ENDPOINT", "").strip()

    # Helper to check if dataset repo is already cached
    def _get_cached_path(repo_id_):
        try:
            return snapshot_download(
                repo_id=repo_id_, repo_type="dataset", local_files_only=True
            )
        except FileNotFoundError:
            return None

    # Private mirror case
    if hf_endpoint:
        if streaming:
            logger.warning(
                "Streaming mode not compatible with private mirror. Falling back to download mode."
            )
        local_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
        return load_dataset(local_path, *args, **kwargs)

    # Public case
    local_path = _get_cached_path(repo_id)
    if local_path is not None and streaming is True:
        # Even though streaming mode: rely on local files if already downloaded
        logger.info("Dataset found in cache. Loading from disk instead of streaming.")
        return load_dataset(local_path, *args, **kwargs)

    return load_dataset(repo_id, streaming=streaming, *args, **kwargs)