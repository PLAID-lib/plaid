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

# %% [markdown]
# ### Add inputs / outputs to a Problem Definition

# %%
# Add unique input and output scalar variables
problem.add_input_scalar_name("in_scalar")
problem.add_output_scalar_name("out_scalar")

# Add list of input and output scalar variables
problem.add_input_scalars_names(["in_scalar2", "in_scalar3"])
problem.add_output_scalars_names(["out_scalar2"])

print()
print(f"{problem.get_input_scalars_names() = }")
print(
    f"{problem.get_output_scalars_names() = }",
)

# %%
# Add unique input and output field variables
problem.add_input_field_name("in_field")
problem.add_output_field_name("out_field")

# Add list of input and output field variables
problem.add_input_fields_names(["in_field2", "in_field3"])
problem.add_output_fields_names(["out_field2"])

print()
print(f"{problem.get_input_fields_names() = }")
print(
    f"{problem.get_output_fields_names() = }",
)

# %%
# Add unique input and output timeseries variables
problem.add_input_timeseries_name("in_timeseries")
problem.add_output_timeseries_name("out_timeseries")

# Add list of input and output timeserie variables
problem.add_input_timeseries_names(["in_timeseries2", "in_timeseries3"])
problem.add_output_timeseries_names(["out_timeseries2"])

print()
print(f"{problem.get_input_timeseries_names() = }")
print(
    f"{problem.get_output_timeseries_names() = }",
)

# %%
# Add unique input and output mesh variables
problem.add_input_mesh_name("in_mesh")
problem.add_output_mesh_name("out_mesh")

# Add list of input and output meshe variables
problem.add_input_meshes_names(["in_mesh2", "in_mesh3"])
problem.add_output_meshes_names(["out_mesh2"])

print()
print(f"{problem.get_input_meshes_names() = }")
print(
    f"{problem.get_output_meshes_names() = }",
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
print()
print(f"{problem.get_task() = }")

# %% [markdown]
# ### Set Problem Definition split

# %%
# Init an empty Dataset
dataset = Dataset()
print()
print(f"{dataset = }")

# Add Samples
dataset.add_samples([Sample(), Sample(), Sample(), Sample()])
print()
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
print()
print(f"{split = }")

# %%
problem.set_split(split)
print()
print(f"{problem.get_split() = }")

# %% [markdown]
# ### Retrieves Problem Definition split indices

# %%
# Get all split indices
print()
print(f"{problem.get_all_indices() = }")

# %% [markdown]
# ### Filter Problem Definition inputs / outputs by name

# %%
print()
print(
    f"{problem.filter_input_scalars_names(['in_scalar', 'in_scalar3', 'in_scalar5']) = }"
)
print(
    f"{problem.filter_output_scalars_names(['out_scalar', 'out_scalar3', 'out_scalar5']) = }"
)
print()
print(f"{problem.filter_input_fields_names(['in_field', 'in_field3', 'in_field5']) = }")
print(
    f"{problem.filter_output_fields_names(['out_field', 'out_field3', 'out_field5']) = }"
)
print()
print(
    f"{problem.filter_input_timeseries_names(['in_timeseries', 'in_timeseries3', 'in_timeseries5']) = }"
)
print(
    f"{problem.filter_output_timeseries_names(['out_timeseries', 'out_timeseries3', 'out_timeseries5']) = }"
)
print()
print(f"{problem.filter_input_meshes_names(['in_mesh', 'in_mesh3', 'in_mesh5']) = }")
print(
    f"{problem.filter_output_meshes_names(['out_mesh', 'out_mesh3', 'out_mesh5']) = }"
)

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
print()
print(f"saving path: {pb_def_save_fname}")

problem._save_to_dir_(pb_def_save_fname)

# %% [markdown]
# ### Load a ProblemDefinition from a directory via initialization

# %%
problem = ProblemDefinition(pb_def_save_fname)
print()
print(problem)

# %% [markdown]
# ### Load from a directory via the ProblemDefinition class

# %%
problem = ProblemDefinition.load(pb_def_save_fname)
print()
print(problem)

# %% [markdown]
# ### Load from a directory via a Dataset instance

# %%
problem = ProblemDefinition()
problem._load_from_dir_(pb_def_save_fname)
print()
print(problem)
