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

from plaid.containers.sample import Sample


class _HFToPlaidSampleConverter(object):
    """Class to convert a huggingface dataset sample to a plaid sample."""

    def __init__(self, ds: datasets.Dataset):
        """Initialization.

        Args:
            ds (datasets.Dataset): Hugging Face dataset.
        """
        self.ds = ds

    def __call__(
        self, sample_id: int
    ):  # pragma: no cover  (not reported with multiprocessing)
        """Convert a single sample from the huggingface dataset to a plaid sample."""
        return Sample.model_validate(pickle.loads(self.ds[sample_id]["sample"]))


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
