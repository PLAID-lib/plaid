"""Examples for PLAID `Sample` objects."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
from plaid import Sample
from plaid.examples.dataset import datasets
from plaid.examples.config import _HF_REPOS


class _LazySamples:
    """Lazy-loaded example samples for PLAID."""

    def __init__(self):
        self._cache:dict[str, Sample] = {}

    def _load_dataset(
        self, ex_name: str) -> Sample:

        if ex_name in self._cache:
            return self._cache[ex_name]

        sample = getattr(datasets, ex_name)[0]
        self._cache[ex_name] = sample

        return sample

def _make_example(ex_name):
    def prop(self):
        return self._load_dataset(ex_name)
    return property(prop)

# Generate properties
for ex_name in _HF_REPOS.keys():
    setattr(_LazySamples, ex_name, _make_example(ex_name))

# Generate class
samples = _LazySamples()
