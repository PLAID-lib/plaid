"""Huggingface private helpers."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import pickle
from pathlib import Path
from typing import Any

import datasets
from datasets import load_from_disk
from pydantic import ValidationError

from plaid import Sample
from plaid.containers.features import SampleMeshes, SampleScalars


def _to_plaid_sample(hf_sample: dict[str, Any]) -> Sample:
    """Convert a Hugging Face dataset sample (pickle) to a plaid :ref:`Sample`.

    If the sample is not valid, it tries to build it from its components.
    If it still fails because of a missing key, it raises a KeyError.
    """
    try:
        # Try to validate the sample
        return Sample.model_validate(hf_sample)
    except ValidationError:
        # If it fails, try to build the sample from its components
        try:
            scalars = SampleScalars(scalars=hf_sample["scalars"])
            meshes = SampleMeshes(
                meshes=hf_sample["meshes"],
                mesh_base_name=hf_sample.get("mesh_base_name"),
                mesh_zone_name=hf_sample.get("mesh_zone_name"),
                links=hf_sample.get("links"),
                paths=hf_sample.get("paths"),
            )
            sample = Sample(
                path=hf_sample.get("path"),
                meshes=meshes,
                scalars=scalars,
                time_series=hf_sample.get("time_series"),
            )
            return Sample.model_validate(sample)
        except KeyError as e:
            raise KeyError(f"Missing key {e!s} in HF data.") from e


class _HFToPlaidSampleConverter:
    """Class to convert a Hugging Face dataset sample to a plaid :ref:`Sample`."""

    def __init__(self, ds: datasets.Dataset):
        self.ds = ds

    def __call__(self, sample_id: int) -> "Sample":  # pragma: no cover
        data = pickle.loads(self.ds[sample_id]["sample"])
        return _to_plaid_sample(data)


class _HFShardToPlaidSampleConverter(object):
    """Class to convert a huggingface dataset sample shard to a plaid sample."""

    def __init__(self, shard_path: Path):
        """Initialization.

        Args:
            shard_path (Path): path of the shard.
        """
        self.ds = load_from_disk(shard_path.as_posix())

    def __call__(
        self, sample_id: int
    ):  # pragma: no cover (not reported with multiprocessing)
        """Convert a sample shard from the huggingface dataset to a plaid sample."""
        sample = self.ds[sample_id]
        return Sample.model_validate(pickle.loads(sample["sample"]))
