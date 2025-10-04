"""Huggingface private helpers."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from pathlib import Path
from typing import Union

import datasets
from datasets import load_from_disk

from plaid import Sample
from plaid.bridges.huggingface_bridge import binary_to_plaid_sample


class _HFToPlaidSampleConverter:
    """Class to convert a Hugging Face dataset sample to a plaid :class:`Sample <plaid.containers.sample.Sample>`."""

    def __init__(self, hf_ds: Union[datasets.Dataset, datasets.DatasetDict]):
        self.hf_ds = hf_ds

    def __call__(self, sample_id: int) -> Sample:  # pragma: no cover
        return binary_to_plaid_sample(self.hf_ds[sample_id])


class _HFShardToPlaidSampleConverter:
    """Class to convert a huggingface dataset sample shard to a plaid :class:`Sample <plaid.containers.sample.Sample>`."""

    def __init__(self, shard_path: Path):
        """Initialization.

        Args:
            shard_path (Path): path of the shard.
        """
        self.hf_ds: Union[datasets.Dataset, datasets.DatasetDict] = load_from_disk(
            shard_path.as_posix()
        )

    def __call__(
        self, sample_id: int
    ) -> Sample:  # pragma: no cover (not reported with multiprocessing)
        """Convert a sample shard from the huggingface dataset to a plaid :class:`Sample <plaid.containers.sample.Sample>`."""
        return binary_to_plaid_sample(self.hf_ds[sample_id])
