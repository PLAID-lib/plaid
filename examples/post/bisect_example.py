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
# # Bisect Plot Examples
#
# ## Introduction
# This notebook explains the use case of the `prepare_datasets`, and `plot_bisect` functions from the Plaid library. The function is used to generate bisect plots for different scenarios using file paths and PLAID objects.
#

# %%
# Importing Required Libraries
from pathlib import Path
import os

from plaid import Dataset
from plaid.post.bisect import plot_bisect, prepare_datasets
from plaid import ProblemDefinition


# %%
# Setting up Directories
try:
    dataset_directory = Path(__file__).parent.parent.parent / "tests" / "post"
except NameError:
    dataset_directory = Path("..") / ".." / ".." / ".." / "tests" / "post"

# %% [markdown]
# ## Prepare Datasets for comparision
#
# Assuming you have reference and predicted datasets, and a problem definition, The `prepare_datasets` function is used to obtain output scalars for subsequent analysis.
#

# %%
# Load PLAID datasets and problem metadata objects
ref_ds = Dataset(dataset_directory / "dataset_ref")
pred_ds = Dataset(dataset_directory / "dataset_near_pred")
problem = ProblemDefinition(dataset_directory / "problem_definition")

# Get output scalars from reference and prediction dataset
ref_out_scalars, pred_out_scalars, out_scalars_names = prepare_datasets(
    ref_ds, pred_ds, problem, verbose=True
)

print(f"{out_scalars_names = }\n")

# %%
# Get output scalar
key = out_scalars_names[0]

print(f"KEY '{key}':\n")
print(f"ID{' ' * 5}--REF_out_scalars--{' ' * 7}--PRED_out_scalars--")

# Print output scalar values for both datasets
index = 0
for item1, item2 in zip(ref_out_scalars[key], pred_out_scalars[key]):
    print(
        f"{str(index).ljust(2)}  |  {str(item1).ljust(20)}  |   {str(item2).ljust(20)}"
    )
    index += 1

# %% [markdown]
# ## Plotting with File Paths
#
# Here, we load the datasets and problem metadata from file paths and use the `plot_bisect` function to generate a bisect plot for a specific scalar, in this case, "scalar_2."

# %%
print("=== Plot with file paths ===")

# Load PLAID datasets and problem metadata from files
ref_path = dataset_directory / "dataset_ref"
pred_path = dataset_directory / "dataset_pred"
problem_path = dataset_directory / "problem_definition"

# Using file paths to generate bisect plot on feature_2
plot_bisect(ref_path, pred_path, problem_path, "feature_2", "differ_bisect_plot")

# %% [markdown]
# ## Plotting with PLAID
#
# In this section, we demonstrate how to use PLAID objects directly to generate a bisect plot. This can be advantageous when working with PLAID datasets in memory.

# %%
print("=== Plot with PLAID objects ===")

# Load PLAID datasets and problem metadata objects
ref_path = Dataset(dataset_directory / "dataset_ref")
pred_path = Dataset(dataset_directory / "dataset_pred")
problem_path = ProblemDefinition(dataset_directory / "problem_definition")

# Using PLAID objects to generate bisect plot on feature_2
plot_bisect(ref_path, pred_path, problem_path, "feature_2", "equal_bisect_plot")

# %% [markdown]
# ## Mixing with Scalar Index and Verbose
#
# In this final section, we showcase a mix of file paths and PLAID objects, incorporating a scalar index and enabling the verbose option when generating a bisect plot. This can provide more detailed information during the plotting process.

# %%
print("=== Mix with scalar index and verbose ===")

# Mix
ref_path = dataset_directory / "dataset_ref"
pred_path = dataset_directory / "dataset_near_pred"
problem_path = ProblemDefinition(dataset_directory / "problem_definition")

# Using scalar index and verbose option to generate bisect plot
scalar_index = 0
plot_bisect(
    ref_path,
    pred_path,
    problem_path,
    scalar_index,
    "converge_bisect_plot",
    verbose=True,
)

os.remove("converge_bisect_plot.png")
os.remove("differ_bisect_plot.png")
os.remove("equal_bisect_plot.png")