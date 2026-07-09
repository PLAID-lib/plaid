"""PLAID package public API."""

from .containers.sample import Sample
from .containers.utils import get_number_of_samples, get_sample_ids
from .infos import Infos
from .problem_definition import ProblemDefinition
from .storage import (
    download_from_hub,
    init_from_disk,
    init_streaming_from_hub,
    load_infos_from_disk,
    load_infos_from_hub,
    load_problem_definitions_from_disk,
    load_problem_definitions_from_hub,
    push_local_problem_definitions_to_hub,
    save_problem_definitions_to_disk,
)
from .version import __version__

__all__ = [
    "__version__",
    "get_number_of_samples",
    "get_sample_ids",
    "Sample",
    "ProblemDefinition",
    "Infos",
    "download_from_hub",
    "init_from_disk",
    "init_streaming_from_hub",
    "load_infos_from_disk",
    "load_infos_from_hub",
    "load_problem_definitions_from_disk",
    "load_problem_definitions_from_hub",
    "push_local_problem_definitions_to_hub",
    "save_problem_definitions_to_disk",
]

import logging

logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)
