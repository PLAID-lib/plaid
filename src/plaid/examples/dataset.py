"""Examples for PLAID `Dataset` objects."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
from plaid import Dataset
from plaid.bridges.huggingface_bridge import load_dataset_from_hub, binary_to_plaid_sample
from plaid.examples.config import _HF_REPOS


class _LazyDatasets:
    """
    Lazy-loaded example datasets for PLAID.

    Access datasets lazily: download and convert only on first access, then cache.
    """

    def __init__(self):
        self._cache:dict[str, Dataset] = {}

    def _load_dataset(
        self, ex_name: str, hf_repo: str) -> Dataset:
        """
        Generic helper to lazily load a HuggingFace dataset and convert it to PLAID.

        Args:
            ex_name (str): Example name.
            hf_repo (str): HuggingFace dataset repository name in the PLAID-datasets community.

        Returns:
            Dataset: The PLAID dataset.

        Raises:
            RuntimeError: If the dataset cannot be downloaded or converted.
        """
        if ex_name in self._cache:
            return self._cache[ex_name]

        try:
            ds_stream = load_dataset_from_hub(hf_repo, split="all_samples", streaming=True)
            samples = []
            for _ in range(2):
                hf_sample = next(iter(ds_stream))
                samples.append(binary_to_plaid_sample(hf_sample))
            dataset = Dataset(samples=samples)
            self._cache[ex_name] = dataset
            return dataset
        except Exception as e: # pragma: no cover
            raise RuntimeError(f"Failed to download or convert dataset '{hf_repo}'.") from e


def _make_example(ex_name, hf_repo):
    def prop(self):
        return self._load_dataset(ex_name, hf_repo)
    return property(prop)

# Generate properties
for ex_name, hf_repo in _HF_REPOS.items():
    setattr(_LazyDatasets, ex_name, _make_example(ex_name, hf_repo))

# Generate class
datasets = _LazyDatasets()
