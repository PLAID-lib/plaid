# %% [markdown]
# # Dataset Splitting Examples
#
# This Jupyter Notebook demonstrates the usage of the split module using the PLAID library. It includes examples of:
#
# 1. Initializing a Dataset
# 2. Splitting a Dataset with ratios
# 3. Splitting a Dataset with fixed sizes
# 4. Splitting a Dataset with ratio and fixed Sizes
# 5. Splitting a Dataset with custom split IDs
#
# This example demonstrates the usage of dataset splitting functions to divide a dataset into training, validation, and test sets. It provides examples of splitting the dataset using different methods and configurations.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
import numpy as np

# %%
# Import necessary libraries and functions
from plaid.utils.init_with_tabular import initialize_dataset_with_tabular_data
from plaid.utils.split import split_dataset

# %%
# Print dict util
def dprint(name: str, dictio: dict):
    print(name, '{')
    for key, value in dictio.items():
	    print("    ", key, ':', value)

    print('}')

# %% [markdown]
# ## Section 1: Initialize Dataset
#
# In this section, we create a dataset with random tabular data for testing purposes. The dataset will be used for subsequent splitting.

# %%
# Create a dataset with random tabular data for testing purposes
nb_scalars = 7
nb_samples = 70
tabular_data = {f'scalar_{j}': np.random.randn(nb_samples) for j in range(nb_scalars)}
dataset = initialize_dataset_with_tabular_data(tabular_data)

print(f"{dataset = }")

# %% [markdown]
# ## Section 2: Splitting a Dataset with Ratios
#
# In this section, we split the dataset into training, validation, and test sets using specified ratios. We also have the option to shuffle the dataset during the split process.

# %%
print("# First split")
options = {
    'shuffle': True,
    'split_ratios': {
        'train': 0.8,
        'val': 0.1,
    },
}

split = split_dataset(dataset, options)
dprint("split =", split)

# %% [markdown]
# ## Section 3: Splitting a Dataset with Fixed Sizes
#
# In this section, we split the dataset into training, validation, and test sets with fixed sample counts for each set. We can also choose to shuffle the dataset during the split.

# %%
print("# Second split")
options = {
    'shuffle': True,
    'split_sizes': {
        'train': 14,
        'val': 8,
        'test': 5,
    },
}

split = split_dataset(dataset, options)
dprint("split =", split)

# %% [markdown]
# ## Section 4: Splitting a Dataset with Ratios and Fixed Sizes
#
# In this section, we split the dataset into training, validation, and test sets with fixed sample counts and sample ratios for each set. We can also choose to shuffle the dataset during the split.

# %%
print("# Third split")
options = {
    'shuffle': True,
    'split_ratios': {
        'train': 0.7,
        'test': 0.1,
    },
    'split_sizes': {
        'val': 7
    }
}

split = split_dataset(dataset, options)
dprint("split =", split)

# %% [markdown]
# ## Section 5: Splitting a Dataset with Custom Split IDs
#
# In this section, we split the dataset based on custom sample IDs for each set. We can specify the sample IDs for training, validation, and prediction sets.

# %%

print("# Fourth split")
options = {
    'split_ids': {
        'train': np.arange(20),
        'val': np.arange(30, 60),
        'predict': np.arange(25, 35),
    },
}

split = split_dataset(dataset, options)
dprint("split =", split)


