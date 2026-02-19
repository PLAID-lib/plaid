# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Backend registry for plaid.storage.

This module centralizes backend wiring so reader/writer code can use a single
source of truth for backend capabilities.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import datasets

from plaid.storage.cgns.reader import CGNSDatasetDict
from plaid.storage.hf_datasets.reader import HFDatasetDict
from plaid.storage.zarr.reader import ZarrDatasetDict

from . import cgns, hf_datasets, webdataset, zarr


@dataclass(frozen=True)
class BackendSpec:
    """Backend wiring for storage operations."""

    name: str
    init_from_disk: Callable[
        [Union[str, Path]], Union[CGNSDatasetDict, HFDatasetDict, ZarrDatasetDict]
    ]
    download_from_hub: Callable[..., str]
    init_streaming_from_hub: Callable[
        ..., dict[str, datasets.IterableDataset] | datasets.IterableDatasetDict
    ]
    generate_to_disk: Callable[..., None]
    push_local_to_hub: Callable[..., None]
    configure_dataset_card: Callable[..., None]
    to_var_sample_dict: Optional[Callable[..., dict[str, Any]]]
    sample_to_var_sample_dict: Optional[Callable[..., dict[str, Any]]]


BACKENDS = {
    "cgns": BackendSpec(
        name="cgns",
        init_from_disk=cgns.init_datasetdict_from_disk,
        download_from_hub=cgns.download_datasetdict_from_hub,
        init_streaming_from_hub=cgns.init_datasetdict_streaming_from_hub,
        generate_to_disk=cgns.generate_datasetdict_to_disk,
        push_local_to_hub=cgns.push_local_datasetdict_to_hub,
        configure_dataset_card=cgns.configure_dataset_card,
        to_var_sample_dict=None,
        sample_to_var_sample_dict=None,
    ),
    "hf_datasets": BackendSpec(
        name="hf_datasets",
        init_from_disk=hf_datasets.init_datasetdict_from_disk,
        download_from_hub=hf_datasets.download_datasetdict_from_hub,
        init_streaming_from_hub=hf_datasets.init_datasetdict_streaming_from_hub,
        generate_to_disk=hf_datasets.generate_datasetdict_to_disk,
        push_local_to_hub=hf_datasets.push_local_datasetdict_to_hub,
        configure_dataset_card=hf_datasets.configure_dataset_card,
        to_var_sample_dict=hf_datasets.to_var_sample_dict,
        sample_to_var_sample_dict=hf_datasets.sample_to_var_sample_dict,
    ),
    "zarr": BackendSpec(
        name="zarr",
        init_from_disk=zarr.init_datasetdict_from_disk,
        download_from_hub=zarr.download_datasetdict_from_hub,
        init_streaming_from_hub=zarr.init_datasetdict_streaming_from_hub,
        generate_to_disk=zarr.generate_datasetdict_to_disk,
        push_local_to_hub=zarr.push_local_datasetdict_to_hub,
        configure_dataset_card=zarr.configure_dataset_card,
        to_var_sample_dict=zarr.to_var_sample_dict,
        sample_to_var_sample_dict=zarr.sample_to_var_sample_dict,
    ),
    "webdataset": BackendSpec(
        name="webdataset",
        init_from_disk=webdataset.init_datasetdict_from_disk,
        download_from_hub=webdataset.download_datasetdict_from_hub,
        init_streaming_from_hub=webdataset.init_datasetdict_streaming_from_hub,
        generate_to_disk=webdataset.generate_datasetdict_to_disk,
        push_local_to_hub=webdataset.push_local_datasetdict_to_hub,
        configure_dataset_card=webdataset.configure_dataset_card,
        to_var_sample_dict=webdataset.to_var_sample_dict,
        sample_to_var_sample_dict=webdataset.sample_to_var_sample_dict,
    ),
}


def available_backends() -> list[str]:
    """Return available backend names in stable order."""
    return list(BACKENDS.keys())


def get_backend(name: str) -> BackendSpec:
    """Return the backend spec for a given name."""
    try:
        return BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"backend {name} not among available ones: {available_backends()}"
        ) from exc
