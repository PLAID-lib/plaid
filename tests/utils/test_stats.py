# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np
import pytest

from plaid.utils.stats import OnlineStatistics, Stats

from ..containers.test_dataset import nb_samples, samples  # noqa: F401
from ..containers.test_sample import base_name, zone_name  # noqa: F401

# %% Fixtures


@pytest.fixture()
def np_samples():
    return np.random.randn(400, 7)


@pytest.fixture()
def online_stats():
    return OnlineStatistics()


@pytest.fixture()
def stats():
    return Stats()


# %% Tests


class Test_OnlineStatistics:
    def test__init__(self, online_stats):
        pass

    def test_add_samples(self, online_stats, np_samples):
        online_stats.add_samples(np_samples)

    def test_add_samples_already_present(self, online_stats, np_samples):
        online_stats.add_samples(np_samples)
        online_stats.add_samples(np_samples)

    def test_get_stats(self, online_stats, np_samples):
        online_stats.add_samples(np_samples)
        online_stats.get_stats()


class Test_Stats:
    def test__init__(self, stats):
        pass

    def test_add_samples(self, stats, samples):
        stats.add_samples(samples)

    def test_get_stats(self, stats, samples):
        stats.add_samples(samples)
        stats.get_stats()
