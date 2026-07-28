"""Callback contracts for the storage API."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..containers.sample import Sample


@dataclass(frozen=True)
class SampleCallbackContext:
    """Sample and metadata provided to a post-write callback."""

    sample: Sample
    split_name: str
    index: int
    path: Path


SampleCallback = Callable[[SampleCallbackContext], None]
