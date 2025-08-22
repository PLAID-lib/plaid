# %% [markdown]
# # Scikit-Learn Wrapper Examples
#
# This Jupyter Notebook demonstrates various use cases for the Scikit-Learn Wrapper classes, including:
#
# 1. Wrapping Scikit-Learn blocks to use PLAID Datasets as input-output
# 2. Adding them to a Scikit-Learn Pipeline
# 3. Fitting the wrapped blocks or the full Pipeline
# 4. Using it to predict
# 5. Saving and Loading the wrapped blocks or the full Pipeline
#
# This notebook provides detailed examples of using the Scikit-Learn Wrapper class to manage data, Samples, and information within a PLAID Scikit-Learn Wrapper. It is intended for documentation purposes and familiarization with the PLAID library.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
from pathlib import Path

import numpy as np

# %%
# Import necessary libraries and functions
from sklearn.decomposition import PCA

from plaid.containers.dataset import Dataset
from plaid.utils.init_with_tabular import initialize_dataset_with_tabular_data
from plaid.wrappers.sklearn import WrappedSklearnTransform

# %% [markdown]
# ## Section 1: Initializing an Empty Scikit-Learn Wrapper and Samples construction
#
# This section demonstrates how to initialize an empty Scikit-Learn Wrapper and handle Samples.

# %% [markdown]
# ### Initialize an empty Scikit-Learn Wrapper

# %%
print("#---# Empty Scikit-Learn Wrapper")
pca = WrappedSklearnTransform(PCA(n_components=8))
print(f"{pca=}")

# %% [markdown]
# ### Build a random representative Dataset
NB_SAMPLES=11
NB_POINTS=237
DIM=3
NB_SCALARS = 4
dataset = initialize_dataset_with_tabular_data({f'scalar_{i}':np.random.randn(NB_SAMPLES) for i in range(NB_SCALARS)})
for sample_id in range(len(dataset)):
    sample = dataset[sample_id]
    sample.init_base(topological_dim=DIM, physical_dim=DIM)
    sample.init_zone(zone_shape=np.array([0,0,0]))
    sample.set_nodes(np.random.rand(NB_POINTS, DIM))

# %% [markdown]
# ### Load a Dataset
print("#---# Load a Dataset")
dataset_path = Path('../../tests/containers/dataset')
dataset = Dataset(dataset_path)
print(f"{dataset=}")

sample = dataset[0]
for i_sample,sample in enumerate(dataset):
    print(f" - {i_sample} -> {sample=}")

    print(f"    - nodes -> {sample.get_nodes().shape=} | {sample.get_nodes().dtype}")

    print(f"    - elements -> keys:{sample.get_elements().keys()}")
    for k,v in sample.get_elements().items():
        print(f"       - {k} -> shape:{v.shape} | dtype:{v.dtype}")

    print(f"    - {sample.get_scalar_names()=}")
    for sn in sample.get_scalar_names():
        print(f"       - {sn} -> {sample.get_scalar(sn)=}")

    print(f"    - {sample.get_field_names()=}")
    for fn in sample.get_field_names():
        print(f"       - {fn} -> shape:{sample.get_field(fn).shape} | dtype:{sample.get_field(fn).dtype}")
