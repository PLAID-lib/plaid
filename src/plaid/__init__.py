"""PLAID package public API."""

import logging

from beartype.claw import beartype_this_package  # <-- boilerplate for victory

beartype_this_package()

from .containers.sample import Sample  # noqa: E402
from .containers.utils import get_number_of_samples, get_sample_ids  # noqa: E402
from .infos import Infos  # noqa: E402
from .problem_definition import ProblemDefinition  # noqa: E402
from .storage import (  # noqa: E402
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
from .version import __version__  # noqa: E402

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


logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)
