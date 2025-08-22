# %% [markdown]
# # Plaid Bisect Plot Use Case
#
# ## Introduction
# This notebook explains the use case of the `plot_bisect` function from the Plaid library. The function is used to generate bisect plots for different scenarios using file paths and PLAID objects.
#

# %%
# Importing Required Libraries
from pathlib import Path
from typing import Union

from plaid.containers.dataset import Dataset
from plaid.post.bisect import plot_bisect, prepare_datasets
from plaid.problem_definition import ProblemDefinition


# %%
def get_project_root(path: Union[str, Path], index: int=3) -> Path:
    """Find the project root path.

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

# Using file paths to generate bisect plot on scalar_2
plot_bisect(ref_path, pred_path, problem_path, "scalar_2", "differ_bisect_plot")

# %% [markdown]
# ## Plotting with PLAID
#
# In this section, we demonstrate how to use PLAID objects directly to generate a bisect plot. This can be advantageous when working with PLAID datasets in memory.

# %%
print("=== Plot with PLAID objects ===")

# Load PLAID datasets and problem metadata objects
ref_path = Dataset(dataset_directory / "dataset_ref")
pred_path = Dataset(dataset_directory / "dataset_ref")
problem_path = ProblemDefinition(dataset_directory / "problem_definition")

# Using PLAID objects to generate bisect plot on scalar_2
plot_bisect(ref_path, pred_path, problem_path, "scalar_2", "equal_bisect_plot")

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

# %%
# Move generated files to post/ directory
import shutil

shutil.move(
    current_directory / "differ_bisect_plot.png",
    current_directory / "post" / "differ_bisect_plot.png",
)
shutil.move(
    current_directory / "equal_bisect_plot.png",
    current_directory / "post" / "equal_bisect_plot.png",
)
shutil.move(
    current_directory / "converge_bisect_plot.png",
    current_directory / "post" / "converge_bisect_plot.png",
)
