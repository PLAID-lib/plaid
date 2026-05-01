"""PLAID package public API."""

#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from .version import __version__
from .containers.sample import Sample
from .problem_definition import ProblemDefinition
from .containers.dataset import Dataset
from .containers.utils import get_number_of_samples, get_sample_ids

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
