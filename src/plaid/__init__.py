"""PLAID package public API."""
from .containers.dataset import Dataset
from .containers.sample import Sample
from .containers.utils import get_number_of_samples, get_sample_ids
from .problem_definition import ProblemDefinition
from .version import __version__

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
