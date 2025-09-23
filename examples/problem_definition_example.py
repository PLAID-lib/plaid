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
# # Problem Definition Examples
#
# This Jupyter Notebook demonstrates the usage of the ProblemDefinition class for defining machine learning problems using the PLAID library. It includes examples of:
#
# 1. Initializing an empty ProblemDefinition
# 2. Configuring problem characteristics and retrieve data
# 3. Saving and loading problem definitions
#
# This notebook provides examples of using the ProblemDefinition class to define machine learning problems, configure characteristics, and save/load problem definitions.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
from pathlib import Path

import numpy as np

# %%
# Import necessary libraries and functions
from plaid import Dataset, Sample
from plaid import ProblemDefinition
from plaid.utils.split import split_dataset
from plaid.types import FeatureIdentifier

# %% [markdown]
# ## Section 1: Initializing an Empty ProblemDefinition
#
# This section demonstrates how to initialize a Problem Definition and add inputs / outputs.

# %% [markdown]
# ### Initialize and print ProblemDefinition

# %%
print("#---# Empty ProblemDefinition")
problem = ProblemDefinition()
print(f"{problem = }")

# %%
# ### Initialize some feature identifiers
scalar_1_feat_id = FeatureIdentifier({"type":"scalar", "name":"scalar_1"})
scalar_2_feat_id = FeatureIdentifier({"type":"scalar", "name":"scalar_2"})
scalar_3_feat_id = FeatureIdentifier({"type":"scalar", "name":"scalar_3"})
field_1_feat_id = FeatureIdentifier({"type":"field", "name":"field_1", "base_name":"Base_2_2"})
field_2_feat_id = FeatureIdentifier({"type":"field", "name":"field_2", "base_name":"Base_2_2", "location":"Vertex"})

# %% [markdown]
# ### Add inputs / outputs to a Problem Definition

# %%
# Add unique input and output feature identifiers
problem.add_in_feature_identifier(scalar_1_feat_id)
problem.add_out_feature_identifier(scalar_2_feat_id)

# Add list of input and output feature identifiers
problem.add_in_features_identifiers([scalar_3_feat_id, field_1_feat_id])
problem.add_out_features_identifiers([field_2_feat_id])

print(f"{problem.get_in_features_identifiers() = }")
print(
    f"{problem.get_out_features_identifiers() = }",
)

# %% [markdown]
# ## Section 2: Configuring Problem Characteristics and retrieve data
#
# This section demonstrates how to handle and configure ProblemDefinition objects and access data.

# %% [markdown]
# ### Set Problem Definition task

# %%
# Set the task type (e.g., regression)
problem.set_task("regression")
print(f"{problem.get_task() = }")

# %% [markdown]
# ### Set Problem Definition split

# %%
# Init an empty Dataset
dataset = Dataset()
print(f"{dataset = }")

# Add Samples
dataset.add_samples([Sample(), Sample(), Sample(), Sample()])
print(f"{dataset = }")

# %%
# Set startegy options for the split
options = {
    "shuffle": False,
    "split_sizes": {
        "train": 2,
        "val": 1,
    },
}

split = split_dataset(dataset, options)
print(f"{split = }")

# %%
problem.set_split(split)
print(f"{problem.get_split() = }")

# %% [markdown]
# ### Retrieves Problem Definition split indices

# %%
# Get all split indices
print(f"{problem.get_all_indices() = }")

# %% [markdown]
# ### Filter Problem Definition inputs / outputs by feature identifiers

# %%
all_feature_ids = [scalar_1_feat_id, scalar_2_feat_id, scalar_3_feat_id, field_1_feat_id, field_2_feat_id]
print(f"{problem.filter_in_features_identifiers(all_feature_ids) = }")
print(f"{problem.filter_out_features_identifiers(all_feature_ids) = }")

# %% [markdown]
# ## Section 3: Saving and Loading Problem Definitions
#
# This section demonstrates how to save and load a Problem Definition from a directory.

# %% [markdown]
# ### Save a Problem Definition to a directory

# %%
test_pth = Path(f"/tmp/test_safe_to_delete_{np.random.randint(low=1, high=2_000_000_000)}")
pb_def_save_fname = test_pth / "test"
test_pth.mkdir(parents=True, exist_ok=True)
print(f"saving path: {pb_def_save_fname}")

problem._save_to_dir_(pb_def_save_fname)

# %% [markdown]
# ### Load a ProblemDefinition from a directory via initialization

# %%
problem = ProblemDefinition(pb_def_save_fname)
print(problem)

# %% [markdown]
# ### Load from a directory via the ProblemDefinition class

# %%
problem = ProblemDefinition.load(pb_def_save_fname)
print(problem)

# %% [markdown]
# ### Load from a directory via a Dataset instance

# %%
problem = ProblemDefinition()
problem._load_from_dir_(pb_def_save_fname)
print(problem)
