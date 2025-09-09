"""Examples for PLAID objects."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from plaid.examples.config import _HF_REPOS

AVAILABLE_EXAMPLES = list(_HF_REPOS.keys())

__all__ = ["datasets", "samples", "AVAILABLE_EXAMPLES"]

# Lazy imports to avoid circular dependency
def __getattr__(name):
    if name == "datasets":
        from plaid.examples.dataset import datasets
        return datasets
    if name == "samples":
        from plaid.examples.sample import samples
        return samples
    raise AttributeError(f"module {__name__} has no attribute {name}")