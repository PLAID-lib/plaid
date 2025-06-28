# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np
import pytest

from plaid.containers.sample import Sample
from plaid.utils.stats import OnlineStatistics, Stats

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
def np_samples_6():
    return np.random.randn(50, 1)


@pytest.fixture()
def online_stats():
    return OnlineStatistics()


@pytest.fixture()
def stats():
    return Stats()


@pytest.fixture()
def sample_with_scalar(np_samples_3):
    s = Sample()
    s.add_scalar("foo", float(np_samples_3.mean()))
    return s


@pytest.fixture()
def sample_with_field(np_samples_6):
    s = Sample()
    # 1. Initialize the CGNS tree
    s.init_tree()
    # 2. Create a base and a zone
    s.init_base(topological_dim=3, physical_dim=3)
    s.init_zone(zone_shape=np.array([np_samples_6.shape[0], 0, 0]))
    # 3. Set node coordinates (required for a valid zone)
    s.set_nodes(np.zeros((np_samples_6.shape[0], 3)))
    # 4. Add a field named "bar"
    s.add_field(name="bar", field=np_samples_6)
    return s


@pytest.fixture()
def time_series_data():
    # 10 time steps, 1 feature
    times = np.linspace(0, 1, 10)
    values = np.random.randn(10)
    return times, values


@pytest.fixture()
def time_series_data_of_different_size():
    # 5 time steps, 1 feature
    times = np.linspace(0, 1, 5)
    values = np.random.randn(5)
    return times, values


@pytest.fixture()
def sample_with_time_series(time_series_data):
    s = Sample()
    times, values = time_series_data
    s.add_time_series("ts1", time_sequence=times, values=values)
    return s


@pytest.fixture()
def sample_with_time_series_of_different_size(time_series_data_of_different_size):
    s = Sample()
    times, values = time_series_data_of_different_size
    s.add_time_series("ts1", time_sequence=times, values=values)
    return s


# %% Tests


class Test_OnlineStatistics:
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

    def test_add_samples_4(self, online_stats, np_samples_5):
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

    def test_invalid_input_type(self, online_stats):
        with pytest.raises(TypeError):
            online_stats.add_samples([1, 2, 3])  # List instead of ndarray

    def test_nan_inf_input(self, online_stats):
        with pytest.raises(ValueError):
            online_stats.add_samples(np.array([1, np.nan, 3]))
        with pytest.raises(ValueError):
            online_stats.add_samples(np.array([1, np.inf, 3]))

    def test_merge_stats(self, np_samples_3, np_samples_4, np_samples_6):
        stats1 = OnlineStatistics()
        stats2 = OnlineStatistics()
        stats1.add_samples(np_samples_3)
        stats2.add_samples(np_samples_6)
        n_samples_before = stats1.n_samples
        n_samples_other = stats2.n_samples
        mean_before = stats1.mean.copy()
        other_mean = stats2.mean.copy()
        stats3 = OnlineStatistics()
        stats3.add_samples(np_samples_4)
        # do the merging
        stats1.merge_stats(stats2)
        assert stats1.n_samples == n_samples_before + stats2.n_samples
        print(f"{n_samples_before=}, {n_samples_other=}")
        print(f"{mean_before=}, {other_mean=}")
        expected_mean = (
            mean_before * n_samples_before + other_mean * n_samples_other
        ) / (n_samples_before + n_samples_other)
        print(f"{expected_mean=}")
        print(f"{stats1.mean=}")
        assert np.allclose(stats1.mean, expected_mean)
        # other merging tests
        with pytest.raises(TypeError):
            stats1.merge_stats(0.0)
        stats1.merge_stats(stats3)


class Test_Stats:
    def test__init__(self, stats):
        pass

    def test_add_samples(self, stats, samples):
        stats.add_samples(samples)

    def test_add_dataset(self, stats, dataset):
        stats.add_dataset(dataset)

    def test_get_stats(self, stats, samples):
        stats.add_samples(samples)
        stats.get_stats()

    def test_invalid_input(self, stats):
        with pytest.raises(TypeError):
            stats.add_samples("invalid")

    def test_empty_samples(self, stats):
        stats.add_samples([])
        assert len(stats.get_available_statistics()) == 0

    def test_merge_stats(self, sample_with_scalar, sample_with_field):
        # Create two Stats objects with different samples
        stats1 = Stats()
        stats2 = Stats()
        stats1.add_samples([sample_with_scalar])
        stats2.add_samples([sample_with_field])
        # Merge stats2 into stats1
        stats1.merge_stats(stats2)
        # Both keys should be present
        keys = stats1.get_available_statistics()
        assert "foo" in keys or "bar" in keys
        # Check that statistics are present for merged keys
        for key in keys:
            s = stats1._stats[key]
            assert s.n_samples > 0

    def test_clear_statistics(self, stats, samples):
        stats.add_samples(samples)
        stats.clear_statistics()
        assert len(stats.get_available_statistics()) == 0

    def test_add_samples_with_time_series(
        self, stats, sample_with_time_series, sample_with_time_series_of_different_size
    ):
        stats.add_samples([sample_with_time_series])
        stats.add_samples([sample_with_time_series])
        keys = stats.get_available_statistics()
        assert "time_series/ts1" in keys
        assert "timestamps/ts1" in keys
        stat = stats._stats["time_series/ts1"]
        assert stat.n_samples == 2
        assert stat.n_points == 20
        stats_dict = stat.get_stats()
        assert "min" in stats_dict
        assert "max" in stats_dict
        assert "n_samples" in stats_dict
        assert "mean" in stats_dict
        assert "var" in stats_dict
        assert "std" in stats_dict
        assert stats_dict["mean"].shape == (
            1,
            len(sample_with_time_series.get_time_series("ts1")[1]),
        )
        stats.add_samples([sample_with_time_series_of_different_size])
        assert "time_series/ts1" in keys
        assert "timestamps/ts1" in keys
        stat = stats._stats["time_series/ts1"]
        # 2 samples of size 10 + 1 sample of size 5 -> 25 values
        assert stat.n_samples == 3
        assert stat.n_points == 25
        stats_dict = stat.get_stats()
        assert "min" in stats_dict
        assert "max" in stats_dict
        assert "n_samples" in stats_dict
        assert "mean" in stats_dict
        assert "var" in stats_dict
        assert "std" in stats_dict
        assert stats_dict["mean"].shape == (1, 1)

    def test_merge_stats_with_time_series(self, sample_with_time_series):
        stats1 = Stats()
        stats2 = Stats()
        stats1.add_samples([sample_with_time_series])
        stats2.add_samples([sample_with_time_series])
        print(
            f"{stats1._stats['time_series/ts1'].n_samples=}, {stats2._stats['time_series/ts1'].n_samples=}"
        )
        stats1.merge_stats(stats2)
        keys = stats1.get_available_statistics()
        assert "time_series/ts1" in keys
        assert "timestamps/ts1" in keys
        stat = stats1._stats["time_series/ts1"]
        assert stat.n_samples == 2
