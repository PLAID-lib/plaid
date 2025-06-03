"""This file is used to define fixtures and configurations for pytest."""

from tests.containers.test_dataset import nb_samples, samples
from tests.containers.test_sample import base_name, zone_name

__all__ = ["samples", "nb_samples", "zone_name", "base_name"]
