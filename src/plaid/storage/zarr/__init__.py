"""Package for Zarr storage."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from plaid.storage.zarr.bridge import (
    sample_to_var_sample_dict,
    to_var_sample_dict,
)
from plaid.storage.zarr.reader import (
    download_datasetdict_from_hub,
    init_datasetdict_from_disk,
    init_datasetdict_streaming_from_hub,
)
from plaid.storage.zarr.writer import (
    configure_dataset_card,
    generate_datasetdict_to_disk,
    push_local_datasetdict_to_hub,
)

__all__ = [
    "configure_dataset_card",
    "download_datasetdict_from_hub",
    "generate_datasetdict_to_disk",
    "init_datasetdict_from_disk",
    "init_datasetdict_streaming_from_hub",
    "push_local_datasetdict_to_hub",
    "sample_to_var_sample_dict",
    "to_var_sample_dict",
]
