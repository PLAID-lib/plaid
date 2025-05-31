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

from ..containers.test_dataset import nb_samples, samples, dataset
from ..containers.test_sample import base_name, zone_name

# %% Fixtures


@pytest.fixture()
def np_samples_1():
    return np.random.randn(400, 7)

@pytest.fixture()
def np_samples_2():
    return np.random.randn(20, 5, 7)

@pytest.fixture()
def np_samples_3():
    return np.random.randn(400, 1)

@pytest.fixture()
def np_samples_4():
    return np.random.randn(1, 400)

@pytest.fixture()
def np_samples_5():
    return np.random.randn(400)

@pytest.fixture()
def online_stats():
    return OnlineStatistics()

@pytest.fixture()
def stats():
    return Stats()


# %% Tests

class Test_OnlineStatistics():
    def test__init__(self, online_stats):
        pass

    def test_add_samples_1(self, online_stats, np_samples_1, np_samples_2):
        online_stats.add_samples(np_samples_1)
        online_stats.add_samples(np_samples_2)

    def test_add_samples_2(self, online_stats, np_samples_4, np_samples_5):
        online_stats.min = np_samples_4
        online_stats.add_samples(np_samples_5)

    def test_add_samples_3(self, online_stats, np_samples_3, np_samples_5):
        online_stats.min = np_samples_3
        online_stats.add_samples(np_samples_5)

    def test_add_samples_already_present(self, online_stats, np_samples_1):
        online_stats.add_samples(np_samples_1)
        online_stats.add_samples(np_samples_1)

    def test_add_samples_and_flatten(self, online_stats, np_samples_1, np_samples_2):
        online_stats.add_samples(np_samples_1)
        online_stats.add_samples(np_samples_2)
        online_stats.flatten_array()

    def test_get_stats(self, online_stats, np_samples_1):
        online_stats.add_samples(np_samples_1)
        online_stats.get_stats()


class Test_Stats():
    def test__init__(self, stats):
        pass

    def test_add_samples(self, stats, samples):
        stats.add_samples(samples)

    def test_add_dataset(self, stats, dataset):
        stats.add_dataset(dataset)

    def test_get_stats(self, stats, samples):
        stats.add_samples(samples)
        stats.get_stats()
