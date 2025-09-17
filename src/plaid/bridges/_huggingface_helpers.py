"""Huggingface private helpers."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import pickle
from pathlib import Path

import datasets
from datasets import load_from_disk
from pydantic import ValidationError

from plaid import Sample
from plaid.containers.features import SampleMeshes, SampleScalars


class _HFToPlaidSampleConverter:
    """Class to convert a Hugging Face dataset sample to a plaid :ref:`Sample`."""

    def __init__(self, ds: datasets.Dataset):
        self.ds = ds

    def __call__(self, sample_id: int) -> "Sample":  # pragma: no cover
        data = pickle.loads(self.ds[sample_id]["sample"])

        try:
            # Try to validate the sample
            return Sample.model_validate(data)
        except ValidationError:
            # If it fails, try to build the sample from its components
            try:
                scalars = SampleScalars(scalars=data["scalars"])
                meshes = SampleMeshes(
                    meshes=data["meshes"],
                    mesh_base_name=data.get("mesh_base_name"),
                    mesh_zone_name=data.get("mesh_zone_name"),
                    links=data.get("links"),
                    paths=data.get("paths"),
                )
                sample = Sample(
                    path=data.get("path"),
                    meshes=meshes,
                    scalars=scalars,
                    time_series=data.get("time_series"),
                )
                return Sample.model_validate(sample)
            except KeyError as e:
                raise KeyError(
                    f"Missing key {e!s} in HF payload (sample_id={sample_id})"
                ) from e


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
