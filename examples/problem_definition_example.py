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
# # Problem Definition
#
# This Jupyter Notebook demonstrates the usage of the ProblemDefinition class for defining machine learning problems using the PLAID library. It includes examples of:
#
# 1. Initializing a complete ProblemDefinition
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
from plaid import ProblemDefinition

# %% [markdown]
# ## Section 1: Initializing a ProblemDefinition
#
# This section demonstrates how to initialize a ProblemDefinition and add
# input/output feature identifiers with the current API.

# %% [markdown]
# ### Initialize feature identifiers

# %%
scalar_1_feat_id = "Global/scalar_1"
scalar_2_feat_id = "Global/scalar_2"
scalar_3_feat_id = "Global/scalar_3"
field_1_feat_id = "Base_2_2/Zone/CellCenterFields/field_1"
field_2_feat_id = "Base_2_2/Zone/VertexFields/field_2"

# %%
print("#---# ProblemDefinition")
problem = ProblemDefinition(
    input_features=[scalar_3_feat_id, field_1_feat_id],
    output_features=[field_2_feat_id],
    train_split={"train": [0, 1]},
    test_split={"test": [2, 3]},
)
print(f"{problem = }")

# %% [markdown]
# ### Add inputs / outputs to a Problem Definition

# %%
# Add unique input and output feature identifiers after initialization
#problem.add_input_features(scalar_1_feat_id)
#problem.add_output_features(scalar_2_feat_id)

# or Add list of input and output feature identifiers
problem.add_input_features([scalar_1_feat_id])
problem.add_output_features([scalar_2_feat_id])

print(f"{problem.input_features = }")
print(
    f"{problem.output_features = }",
)

# %% [markdown]
# ## Section 2: Configuring Problem Characteristics and retrieve data
#
# This section demonstrates how to handle and configure ProblemDefinition objects and access data.

# %% [markdown]
# ### Problem Definition identifier
#
# ProblemDefinition has no stored name. When saved in a dataset, its identifier
# is the YAML filename stem or the key in a `dict[str, ProblemDefinition]`.

# %% [markdown]
# ### Set Problem Definition split

# %%
# Current API uses required `train_split` and `test_split` fields.
# Note: each split field currently expects a dictionary with a single entry.
print(f"{problem.train_split = }")
print(f"{problem.test_split = }")

# %%
split_names = [next(iter(problem.train_split)), next(iter(problem.test_split))]
split_indices = {
    next(iter(problem.train_split)): next(iter(problem.train_split.values())),
    next(iter(problem.test_split)): next(iter(problem.test_split.values())),
}
print(f"{split_names = }")
print(f"{split_indices = }")

# %% [markdown]
# ### Show inputs / outputs

# %%
print(f"{problem.input_features = }")
print(f"{problem.output_features = }")

# %% [markdown]
# ## Section 3: Saving and Loading Problem Definitions
#
# This section demonstrates how to save and load a Problem Definition from a directory.

# %% [markdown]
# ### Save a Problem Definition to a YAML file

# %%
test_pth = Path(
    f"/tmp/test_safe_to_delete_{np.random.randint(low=1, high=2_000_000_000)}"
)
pb_def_save_fname = test_pth / "test_problem_definition.yaml"
test_pth.mkdir(parents=True, exist_ok=True)
print(f"saving path: {pb_def_save_fname}")

problem.save_to_file(pb_def_save_fname)

# %% [markdown]
# ### Load a ProblemDefinition from a YAML file

# %%
problem = ProblemDefinition.from_path(pb_def_save_fname)
print(problem)
