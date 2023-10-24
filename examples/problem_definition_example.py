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
import numpy as np
import os

# %%
# Import necessary libraries and functions
from plaid.containers.dataset import Dataset, Sample
from plaid.problem_definition import ProblemDefinition
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
# Add unique input and output variables
problem.add_input('in')
problem.add_output('out')

# Add list of input and output variables
problem.add_inputs(['in2', 'in3'])
problem.add_outputs(['out2'])

print(f"{problem.get_inputs() = }")
print(f"{problem.get_outputs() = }", )

# %% [markdown]
# ## Section 2: Configuring Problem Characteristics and retrieve data
#
# This section demonstrates how to handle and configure ProblemDefinition objects and access data.

# %% [markdown]
# ### Set Problem Definition task

# %%
# Set the task type (e.g., regression)
problem.set_task('regression')
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
    'shuffle': False,
    'split_sizes': {
        'train': 2,
        'val': 1,
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
# ### Filter Problem Definition inputs / outputs by name

# %%
print(f"{problem.filter_input_names(['in', 'in3', 'in5']) = }")
print(f"{problem.filter_output_names(['out', 'out3', 'out5']) = }")

# %% [markdown]
# ## Section 3: Saving and Loading Problem Definitions
#
# This section demonstrates how to save and load a Problem Definition from a directory.

# %% [markdown]
# ### Save a Problem Definition to a directory

# %%
test_pth = f"/tmp/test_safe_to_delete_{np.random.randint(1e10, 1e12)}"
pb_def_save_fname = os.path.join(test_pth, 'test')
os.makedirs(test_pth)
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


