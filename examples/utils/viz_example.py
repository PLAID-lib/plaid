# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualization Examples
#
# This notebook demonstrates the visualization capabilities of PLAID for analyzing datasets.
# It covers three main visualization functions:
#
# 1. **scatter_plot**: Visualize how feature values vary across samples
# 2. **pairplot**: Show pairwise relationships between scalar features
# 3. **kdeplot**: Display probability density distributions of features
#
# Each visualization function helps in understanding different aspects of your dataset:
# - Detecting trends and outliers
# - Understanding correlations between features
# - Comparing distributions
#
# **Each section is documented and explained.**

# %%
# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# %%
# Import necessary PLAID classes and visualization functions
from plaid import Dataset, Sample
from plaid.utils.viz import scatter_plot, pairplot, kdeplot

# %% [markdown]
# ## Section 1: Creating a Sample Dataset
#
# First, we'll create a sample dataset with multiple samples containing both scalar
# and field features. This dataset will be used to demonstrate all visualization functions.

# %%
print("#---# Create sample dataset")

# Number of samples to create
n_samples = 50

# Create samples with scalars and fields
samples = []
for i in range(n_samples):
    sample = Sample()

    # Add scalar features with some relationships
    # temperature and pressure are correlated
    temperature = 20 + 5 * np.random.randn() + 0.5 * i
    pressure = 100 + 10 * np.random.randn() + 0.8 * temperature

    # density is independent
    density = 1.2 + 0.2 * np.random.randn()

    # velocity has a trend
    velocity = 10 + 0.1 * i + 2 * np.random.randn()

    sample.add_scalar("temperature", temperature)
    sample.add_scalar("pressure", pressure)
    sample.add_scalar("density", density)
    sample.add_scalar("velocity", velocity)

    # Add a field feature
    sample.init_base(2, 3, "mesh_base")
    zone_shape = np.array([100, 0, 0])
    sample.init_zone(zone_shape, zone_name="zone_1")

    # Create a field with spatial variation
    field_data = np.sin(np.linspace(0, 2 * np.pi, 100)) * (1 + 0.1 * i)
    sample.add_field("displacement", field_data)

    samples.append(sample)

# Create dataset from samples
dataset = Dataset()
dataset.add_samples(samples)

print(f"Created dataset with {len(dataset)} samples")
print(f"Scalar features: {dataset.get_scalar_names()}")
print(f"Field features: {dataset.get_field_names()}")

# %% [markdown]
# ## Section 2: Scatter Plot - Feature vs Sample ID
#
# The scatter_plot function visualizes how feature values change across samples.
# This is useful for detecting trends, outliers, or patterns in your data.

# %% [markdown]
# ### Example 2.1: Plot all scalar features

# %%
print("#---# Scatter plot of all scalar features")

fig = scatter_plot(dataset)
plt.show()

# %% [markdown]
# ### Example 2.2: Plot specific features with customization

# %%
print("#---# Scatter plot of specific features")

fig = scatter_plot(
    dataset,
    feature_names=["temperature", "velocity"],
    figsize=(12, 5),
    alpha=0.6,
    s=50,  # marker size
    c="blue",  # marker color
    title="Temperature and Velocity Trends",
)
plt.show()

# %% [markdown]
# ### Example 2.3: Plot subset of samples

# %%
print("#---# Scatter plot for first 20 samples only")

# Get first 20 sample IDs
sample_ids = dataset.get_sample_ids()[:20]

fig = scatter_plot(
    dataset,
    feature_names=["pressure", "density"],
    sample_ids=sample_ids,
    title="Pressure and Density (First 20 Samples)",
)
plt.show()

# %% [markdown]
# ## Section 3: Pairplot - Relationships Between Features
#
# The pairplot function creates a matrix showing pairwise relationships between features.
# The diagonal shows distributions, while off-diagonal plots show scatter plots.
# This helps identify correlations and multivariate patterns.

# %% [markdown]
# ### Example 3.1: Basic pairplot with all scalars

# %%
print("#---# Pairplot of all scalar features")

fig = pairplot(dataset)
plt.show()

# %% [markdown]
# ### Example 3.2: Pairplot with specific features and KDE on diagonal

# %%
print("#---# Pairplot with KDE on diagonal")

fig = pairplot(
    dataset,
    scalar_names=["temperature", "pressure", "velocity"],
    diag_kind="kde",
    title="Feature Relationships (KDE on diagonal)",
)
plt.show()

# %% [markdown]
# ### Example 3.3: Corner pairplot (lower triangle only)

# %%
print("#---# Corner pairplot")

fig = pairplot(
    dataset,
    scalar_names=["temperature", "pressure", "density"],
    corner=True,
    diag_kind="hist",
    title="Corner Pairplot",
    alpha=0.5,
)
plt.show()

# %% [markdown]
# ## Section 4: KDE Plot - Distribution Comparison
#
# The kdeplot function shows smooth probability density estimates for features.
# Multiple features can be overlaid on the same plot for easy comparison.

# %% [markdown]
# ### Example 4.1: KDE plot of all scalar features

# %%
print("#---# KDE plot of all scalar features")

fig = kdeplot(dataset)
plt.show()

# %% [markdown]
# ### Example 4.2: Compare distributions of specific features

# %%
print("#---# KDE plot comparing specific features")

fig = kdeplot(
    dataset,
    feature_names=["temperature", "velocity"],
    title="Temperature vs Velocity Distributions",
    fill=True,
)
plt.show()

# %% [markdown]
# ### Example 4.3: KDE plot without fill and custom bandwidth

# %%
print("#---# KDE plot with custom styling")

fig = kdeplot(
    dataset,
    feature_names=["pressure", "density"],
    fill=False,
    bw_method="silverman",  # alternative bandwidth method
    linewidth=2.5,
    title="Pressure and Density Distributions (No Fill)",
)
plt.show()

# %% [markdown]
# ### Example 4.4: KDE plot for a single feature

# %%
print("#---# KDE plot for single feature")

fig = kdeplot(
    dataset,
    feature_names=["temperature"],
    title="Temperature Distribution",
    figsize=(8, 5),
)
plt.show()

# %% [markdown]
# ## Section 5: Combining Visualizations
#
# Often, it's useful to combine multiple visualization types to get a complete
# picture of your data.

# %%
print("#---# Create a combined visualization layout")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 10))

# Subplot 1: Scatter plot of temperature
ax1 = plt.subplot(2, 2, 1)
temp_ids = dataset.get_sample_ids()
temp_values = [dataset[sid].get_scalar("temperature") for sid in temp_ids]
ax1.scatter(temp_ids, temp_values, alpha=0.6)
ax1.set_xlabel("Sample ID")
ax1.set_ylabel("Temperature")
ax1.set_title("Temperature Trend")
ax1.grid(True, alpha=0.3)

# Subplot 2: Scatter plot of pressure vs temperature
ax2 = plt.subplot(2, 2, 2)
pressure_values = [dataset[sid].get_scalar("pressure") for sid in temp_ids]
ax2.scatter(temp_values, pressure_values, alpha=0.6)
ax2.set_xlabel("Temperature")
ax2.set_ylabel("Pressure")
ax2.set_title("Pressure vs Temperature")
ax2.grid(True, alpha=0.3)

# Subplot 3: Histograms
ax3 = plt.subplot(2, 2, 3)
ax3.hist(temp_values, bins=15, alpha=0.7, label="Temperature")
ax3.set_xlabel("Value")
ax3.set_ylabel("Frequency")
ax3.set_title("Temperature Distribution (Histogram)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: KDE comparison
ax4 = plt.subplot(2, 2, 4)
from scipy import stats

# Temperature KDE
kde_temp = stats.gaussian_kde(temp_values)
x_temp = np.linspace(min(temp_values), max(temp_values), 100)
ax4.plot(x_temp, kde_temp(x_temp), label="Temperature", linewidth=2)
ax4.fill_between(x_temp, kde_temp(x_temp), alpha=0.3)

# Velocity KDE
velocity_values = [dataset[sid].get_scalar("velocity") for sid in temp_ids]
kde_vel = stats.gaussian_kde(velocity_values)
x_vel = np.linspace(min(velocity_values), max(velocity_values), 100)
ax4.plot(x_vel, kde_vel(x_vel), label="Velocity", linewidth=2)
ax4.fill_between(x_vel, kde_vel(x_vel), alpha=0.3)

ax4.set_xlabel("Value")
ax4.set_ylabel("Density")
ax4.set_title("KDE Comparison")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()