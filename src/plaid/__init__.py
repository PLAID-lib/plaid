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

from .containers.dataset import Dataset
from .containers.sample import Sample
from .containers.utils import get_number_of_samples, get_sample_ids
from .problem_definition import ProblemDefinition

__all__ = [
    "__version__",
    "get_number_of_samples",
    "get_sample_ids",
    "Dataset",
    "Sample",
    "ProblemDefinition",
]

import logging

logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)
