"""WebDataset storage backend for PLAID.

This module provides WebDataset format support for PLAID datasets, enabling tar-based
storage with streaming capabilities and Hugging Face Hub integration.

The WebDataset backend uses tar archives where samples with the same basename belong
together (e.g., sample_000000000.json and sample_000000000.npy). This format is
ideal for streaming large physics datasets and has excellent compatibility with
Hugging Face Hub.

Public API:
    - init_datasetdict_from_disk: Load dataset from local tar files
    - download_datasetdict_from_hub: Download dataset from Hub
    - init_datasetdict_streaming_from_hub: Stream dataset from Hub
    - generate_datasetdict_to_disk: Generate and save dataset to tar archives
    - push_local_datasetdict_to_hub: Upload dataset to Hub
    - configure_dataset_card: Create and push dataset card
    - to_var_sample_dict: Extract variable features from sample
    - sample_to_var_sample_dict: Convert sample to variable sample dict
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from plaid.storage.webdataset.bridge import (
    sample_to_var_sample_dict,
    to_var_sample_dict,
)
from plaid.storage.webdataset.reader import (
    download_datasetdict_from_hub,
    init_datasetdict_from_disk,
    init_datasetdict_streaming_from_hub,
)
from plaid.storage.webdataset.writer import (
    configure_dataset_card,
    generate_datasetdict_to_disk,
    push_local_datasetdict_to_hub,
)

__all__ = [
    # Reader functions
    "init_datasetdict_from_disk",
    "download_datasetdict_from_hub",
    "init_datasetdict_streaming_from_hub",
    # Writer functions
    "generate_datasetdict_to_disk",
    "push_local_datasetdict_to_hub",
    "configure_dataset_card",
    # Bridge functions
    "to_var_sample_dict",
    "sample_to_var_sample_dict",
]
