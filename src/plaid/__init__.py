"""PLAID package public API."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "None"

from .containers.utils import get_number_of_samples, get_sample_ids

__all__ = [
    "__version__",
    "get_number_of_samples",
    "get_sample_ids",
]
