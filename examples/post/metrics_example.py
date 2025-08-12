# %% [markdown]
# # PLAID Metrics Use Case
#
# ## Introduction
# This notebook demonstrates the use case of the `compute_metrics` function from the PLAID library. The function is used to compute metrics for comparing reference and predicted datasets based on a given problem definition.
#

# %%
# Importing Required Libraries
import os
from typing import Union
from pathlib import Path

from plaid.containers.dataset import Dataset
from plaid.post.metrics import compute_metrics, prepare_datasets, pretty_metrics
from plaid.problem_definition import ProblemDefinition


# %%
def get_project_root(path: Union[str, Path], index: int=3) -> Path:
    """Find the project root path

    Args:
        path (Union[str, Path]): Current path of the notebook
        index (int, optional): The number of parents to go back. Defaults to 3.

    Returns:
        Path: The project root path
    """
    path = Path(path)
    if index == 0:
        return path
    return get_project_root(path.parent, index - 1)


# Setting up Directories
current_directory = Path.cwd()
# dataset_directory = get_project_root(current_directory) / "tests" / "post"
dataset_directory = get_project_root(current_directory, 1) / "tests" / "post"

# %% [markdown]
# ## Prepare Datasets for comparision
#
# Assuming you have reference and predicted datasets, and a problem definition, The `prepare_datasets` function is used to obtain output scalars for subsequent analysis.

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
# ## Metrics with File Paths
#
# Here, we load the datasets and problem metadata from file paths and use the `compute_metrics` function to generate metrics for comparison. The resulting metrics are then printed in a structured dictionary format.

# %%
print("=== Metrics with file paths ===")

# Load PLAID datasets and problem metadata file paths
ref_ds = dataset_directory / "dataset_ref"
pred_ds = dataset_directory / "dataset_near_pred"
problem = dataset_directory / "problem_definition"

# Using file paths to generate metrics
metrics = compute_metrics(ref_ds, pred_ds, problem, "first_metrics")

import json

# Print the resulting metrics
print("output dictionary =", json.dumps(metrics, indent=4))

# %% [markdown]
# ## Metrics with PLAID Objects and Verbose
#
# In this section, we demonstrate how to use PLAID objects directly to generate metrics, and the verbose option is enabled to provide more detailed information during the computation.

# %%
print("=== Metrics with PLAID objects and verbose ===")

# Load PLAID datasets and problem metadata objects
ref_ds = Dataset(dataset_directory / "dataset_ref")
pred_ds = Dataset(dataset_directory / "dataset_pred")
problem = ProblemDefinition(dataset_directory / "problem_definition")

# Pretty print activated with verbose mode
metrics = compute_metrics(ref_ds, pred_ds, problem, "second_metrics", verbose=True)

# %% [markdown]
# ## Print metrics in a beautiful way
#
# Finally, in this last section, we showcase a way to print metrics in a more aesthetically pleasing format using the `pretty_metrics` function. The provided dictionary is an example structure for representing metrics, and the function enhances the readability of the metrics presentation. (it is used by `compute_metrics` when verbose mode is activated)

# %%
dictionary: dict = {
    "RMSE:": {
        "train": {"scalar_1": 0.12345, "scalar_2": 0.54321},
        "test": {"scalar_1": 0.56789, "scalar_2": 0.98765},
    }
}

pretty_metrics(dictionary)

# %%
# Move generated files to post/ directory
import shutil

shutil.move(
    current_directory / "first_metrics.yaml",
    current_directory / "post" / "first_metrics.yaml",
)
shutil.move(
    current_directory / "second_metrics.yaml",
    current_directory / "post" / "second_metrics.yaml",
)
