# %% [markdown]
# # Initializing a Dataset with Tabular Data
#
# 1. Initializing a Dataset with Tabular Data:
# - Generate random tabular data for multiple scalars.
# - Initialize a dataset with the tabular data.
#
# 2. Accessing and Manipulating Data in the Dataset:
# - Retrieve and print the dataset and specific samples.
# - Access and display the value of a particular scalar within a sample.
# - Retrieve tabular data from the dataset based on scalar names.
#
# This example demonstrates how to initialize a dataset with tabular data, access specific samples, retrieve scalar values, and extract tabular data based on scalar names.

# %%
# Import required libraries
import numpy as np

# %%
# Import necessary libraries and functions
from plaid.utils.init_with_tabular import initialize_dataset_with_tabular_data

# %%
# Print dict util
def dprint(name: str, dictio: dict):
    print(name, '{')
    for key, value in dictio.items():
	    print("    ", key, ':', value)

    print('}')

# %% [markdown]
# ## Section 1: Initializing a Dataset with Tabular Data

# %%
# Generate random tabular data for multiple scalars
nb_scalars = 7
nb_samples = 10
names = [f"scalar_{j}" for j in range(nb_scalars)]

tabular_data = {}
for name in names:
    tabular_data[name] = np.random.randn(nb_samples)

dprint("tabular_data", tabular_data)

# %%
# Initialize a dataset with the tabular data
dataset = initialize_dataset_with_tabular_data(tabular_data)
print("Initialized Dataset: ", dataset)

# %% [markdown]
# ## Section 2: Accessing and Manipulating Data in the Dataset

# %%
# Retrieve and print the dataset and specific samples
sample_1 = dataset[1]
print(f"{sample_1 = }")


# %%
# Access and display the value of a particular scalar within a sample
scalar_value = sample_1.get_scalar("scalar_0")
print("Scalar 'scalar_0' in Sample 1:", scalar_value)

# %%
# Retrieve tabular data from the dataset based on scalar names
scalar_names = ["scalar_1", "scalar_3", "scalar_5"]
tabular_data_subset = dataset.get_scalars_to_tabular(scalar_names)
print("Tabular Data Subset for Scalars 1, 3, and 5:")
dprint("tabular_data_subset", tabular_data_subset)


