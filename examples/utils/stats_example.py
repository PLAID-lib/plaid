# %% [markdown]
# # Statistics Calculation Examples
#
# 1. OnlineStatistics Class:
# - Initialize an OnlineStatistics object.
# - Calculate statistics for an empty dataset.
# - Add the first batch of samples and update statistics.
# - Add the second batch of samples and update statistics.
# - Combine and recompute statistics for all samples.
#
# 2. Stats Class:
# - Initialize a Stats object to collect statistics.
# - Create and add samples with scalar and field data.
# - Retrieve and display the calculated statistics.
# - Add more samples with varying field sizes and update statistics.
# - Retrieve and display the updated statistics.
#
# This notebook provides examples of using the OnlineStatistics and Stats classes to compute statistics from sample data, including scalars and fields. It demonstrates the functionality and usage of these classes.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
import numpy as np
import rich

# %%
# Import necessary libraries and functions
from plaid.containers.sample import Sample
from plaid.utils.stats import OnlineStatistics, Stats


# %%
def sprint(stats: dict):
    print("Stats:")
    for k,v in stats.items():
        print(f" - {k:>9} -> {v}")


# %% [markdown]
# ## Section 1: OnlineStatistics Class
#
# In this section, we demonstrate the usage of the OnlineStatistics class. We initialize an OnlineStatistics object and calculate statistics for an empty dataset. Then, we add the first and second batches of samples and update the statistics. Finally, we combine and recompute statistics for all samples.

# %% [markdown]
# ### Initialize and empty OnlineStatistics

# %%
print("#---# Initialize OnlineStatistics")
stats_computer = OnlineStatistics()
stats = stats_computer.get_stats()

sprint(stats)

# %% [markdown]
# ### Add sample batches

# %%
# First batch of samples
first_batch_samples = 3.0 * np.random.randn(100, 3) + 10.0
print()
print(f"=== {first_batch_samples.shape = }")

stats_computer.add_samples(first_batch_samples)
stats = stats_computer.get_stats()

sprint(stats)

# %%
second_batch_samples = 10.0 * np.random.randn(1000, 3) - 1.0
print()
print(f"=== {second_batch_samples.shape = }")

stats_computer.add_samples(second_batch_samples)
stats = stats_computer.get_stats()

sprint(stats)

# %% [markdown]
# ### Combine and recompute statistics

# %%
total_samples = np.concatenate((first_batch_samples, second_batch_samples), axis=0)
print()
print(f"=== {total_samples.shape = }")

new_stats_computer = OnlineStatistics()
new_stats_computer.add_samples(total_samples)
stats = new_stats_computer.get_stats()

sprint(stats)

# %% [markdown]
# ## Section 2: Stats Class
#
# In this section, we explore the Stats class. We initialize a Stats object to collect statistics, create and add samples with scalar and field data. We retrieve and display the calculated statistics. We also add more samples with varying field sizes and update the statistics, followed by retrieving and displaying the updated statistics.

# %% [markdown]
# ### Initalize an empty Stats object

# %%
print("#---# Initialize Stats")
stats = Stats()
print(f"{stats.get_stats() = }")

# %% [markdown]
# ### Feed Stats with Samples

# %%
print("#---# Feed Stats with samples")

# Init 11 samples
nb_samples = 11
samples = [Sample() for _ in range(nb_samples)]

spatial_shape_max = 5
#
for sample in samples:
    sample.scalars.add("test_scalar", np.random.randn())
    sample.scalars.add("test_ND_scalar", np.random.randn(3))
    sample.init_base(2, 3,)
    zone_shape = np.array([0, 0, 0])
    sample.init_zone(zone_shape)
    sample.add_field("test_field", np.random.randn(spatial_shape_max))
    sample.init_zone(zone_shape, zone_name="test_zone_2")
    sample.add_field("test_field", np.random.randn(spatial_shape_max), zone_name='test_zone_2')
    sample.init_base(2, 3, "test_base_2")
    zone_shape = np.array([0, 0, 0])
    sample.init_zone(zone_shape, zone_name="test_zone_1", base_name="test_base_2")
    sample.add_field("test_field", np.random.randn(spatial_shape_max), base_name="test_base_2")
    sample.init_zone(zone_shape, zone_name="test_zone_2", base_name="test_base_2")
    sample.add_field("test_field", np.random.randn(spatial_shape_max), zone_name='test_zone_2', base_name="test_base_2")

stats.add_samples(samples)

# %% [markdown]
# ### Get and print stats

# %%
rich.print("stats.get_stats():")
rich.print(stats.get_stats())

# %% [markdown]
# ### Feed Stats with more Samples

# %%
nb_samples = 13
# spatial_shape_max = 20
samples = [Sample() for _ in range(nb_samples)]

for sample in samples:
    sample.scalars.add("test_scalar", np.random.randn())
    sample.init_base(2, 3,)
    zone_shape = np.array([0, 0, 0])
    sample.init_zone(zone_shape)
    sample.add_field("test_field", np.random.randn(spatial_shape_max))
    sample.init_zone(zone_shape, zone_name="test_zone_2")
    sample.add_field("test_field", np.random.randn(spatial_shape_max), zone_name='test_zone_2')
    sample.init_base(2, 3, "test_base_2")
    zone_shape = np.array([0, 0, 0])
    sample.init_zone(zone_shape, zone_name="test_zone_1", base_name="test_base_2")
    sample.add_field("test_field", np.random.randn(spatial_shape_max), base_name="test_base_2")
    sample.init_zone(zone_shape, zone_name="test_zone_2", base_name="test_base_2")
    sample.add_field("test_field", np.random.randn(spatial_shape_max), zone_name='test_zone_2', base_name="test_base_2")

    sample.init_base(2, 3, "test_base")
    zone_shape = np.array([0, 0, 0])
    sample.init_zone(zone_shape, zone_name="test_zone", base_name="test_base")
    sample.add_field("test_field_same_size", np.random.randn(spatial_shape_max), zone_name="test_zone", base_name="test_base")
    sample.add_field(
        "test_field",
        np.random.randn(np.random.randint(spatial_shape_max // 2, spatial_shape_max)),
        zone_name="test_zone", base_name="test_base"
    )

stats.add_samples(samples)

# %% [markdown]
# ### Get and print stats

# %%
rich.print("stats.get_stats():")
rich.print(stats.get_stats())
