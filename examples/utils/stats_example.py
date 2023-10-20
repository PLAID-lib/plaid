# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import numpy as np
import rich

from plaid.containers.sample import Sample
from plaid.utils.stats import OnlineStatistics, Stats

# %% Functions


def stats_examples():
    """
    This function shows the usage and functionality of statistics calculation classes.

    Example Usage:

    1. OnlineStatistics Class:
    - Initialize an OnlineStatistics object.
    - Calculate statistics for an empty dataset.
    - Add the first batch of samples and update statistics.
    - Add the second batch of samples and update statistics.
    - Combine and recompute statistics for all samples.

    2. Stats Class:
    - Initialize a Stats object to collect statistics.
    - Create and add samples with scalar and field data.
    - Retrieve and display the calculated statistics.
    - Add more samples with varying field sizes and update statistics.
    - Retrieve and display the updated statistics.

    This function provides examples of utilizing the OnlineStatistics and Stats classes
    to compute statistics from sample data, including scalars and fields. It is intended
    for documentation purposes and familiarization with the PLAID library.
    """
    # %%
    print(f"#---# Initialize OnlineStatistics")
    stats_computer = OnlineStatistics()
    stats = stats_computer.get_stats()
    for k in stats:
        print(" - {} -> {}".format(k, stats[k]))

    # %%
    first_batch_samples = 3.0 * np.random.randn(100, 3) + 10.0

    stats_computer.add_samples(first_batch_samples)
    stats = stats_computer.get_stats()
    for k in stats:
        print(" - {} -> {}".format(k, stats[k]))

    # %%
    second_batch_samples = 10.0 * np.random.randn(1000, 3) - 1.0

    stats_computer.add_samples(second_batch_samples)
    stats = stats_computer.get_stats()
    for k in stats:
        print(" - {} -> {}".format(k, stats[k]))

    # %%
    total_samples = np.concatenate(
        (first_batch_samples, second_batch_samples), axis=0)
    new_stats_computer = OnlineStatistics()
    new_stats_computer.add_samples(total_samples)
    stats = new_stats_computer.get_stats()
    for k in stats:
        print(" - {} -> {}".format(k, stats[k]))

    # %%
    print()
    print(f"#---# Initialize Stats")
    stats = Stats()

    # %%
    print(f"#---# Feed Stats with samples")
    nb_samples = 11
    spatial_shape_max = 20
    samples = [Sample() for _ in range(nb_samples)]
    for sample in samples:
        sample.add_scalar('test_scalar', np.random.randn())
        sample.init_base(2, 3, 'test_base')
        zone_shape = np.array([0, 0, 0])
        sample.init_zone('test_zone', zone_shape)
        sample.add_field('test_field', np.random.randn(spatial_shape_max))

    stats.add_samples(samples)

    # %%
    print(f"#---# Get stats")
    rich.print(f"stats.get_stats():")
    rich.print(stats.get_stats())

    # %%
    print(f"#---# Feed Stats with more samples")
    nb_samples = 11
    spatial_shape_max = 20
    samples = [Sample() for _ in range(nb_samples)]
    for sample in samples:
        sample.add_scalar('test_scalar', np.random.randn())
        sample.init_base(2, 3, 'test_base')
        zone_shape = np.array([0, 0, 0])
        sample.init_zone('test_zone', zone_shape)
        sample.add_field('test_field_same_size', np.random.randn(7))
        sample.add_field(
            'test_field',
            np.random.randn(
                np.random.randint(
                    spatial_shape_max // 2,
                    spatial_shape_max)))

    stats.add_samples(samples)

    # %%
    print(f"#---# Get stats")
    rich.print(f"stats.get_stats():")
    rich.print(stats.get_stats())


# %% Main Script
if __name__ == '__main__':
    stats_examples()
