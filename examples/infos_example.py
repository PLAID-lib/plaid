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
# # Infos
#
# This Jupyter Notebook demonstrates the usage of the Infos class for defining
# dataset metadata using the PLAID library. It includes examples of:
#
# 1. Initializing Infos from structured fields
# 2. Configuring metadata and retrieving data
# 3. Saving and loading infos
#
# This notebook provides examples of using the Infos class to define dataset
# metadata, access entries with typed attributes, and save/load infos.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
from pathlib import Path

import numpy as np

# %%
# Import necessary libraries and classes
from plaid.infos import Infos

# %% [markdown]
# ## Section 1: Initializing Infos
#
# This section demonstrates how to initialize Infos with the current API.

# %% [markdown]
# ### Initialize and print Infos

# %%
print("#---# Infos")
infos = Infos(
    owner="PLAID",
    license="MIT",
    data_production={
        "type": "simulation",
        "physics": "fluid dynamics",
        "simulator": "ExampleSolver",
    },
    data_description="ExampleDescription",
)
print(f"{infos = }")

# %% [markdown]
# ### Print available Infos fields

# %%
Infos.print_available_fields()

# %% [markdown]
# ## Section 2: Modifying Infos and retrieve data
#
# This section demonstrates how to handle Infos objects and access metadata.

# %% [markdown]
# ### Set data description

# %%
infos.data_description = "Example dataset generated for the Infos example."

print(f"{infos.data_description = }")
print(f"{infos.num_samples = }")  # Populated by save_to_disk for saved datasets.
print(f"{infos.storage_backend = }")  # Populated by save_to_disk for saved datasets.

# %% [markdown]
# ### Retrieve data with Pydantic attributes

# %%
print(f"{infos.owner = }")
print(f"{infos.license = }")
print(f"{infos.storage_backend = }")
print(f"{infos.model_dump(exclude_none=True) = }")

# %% [markdown]
# ## Section 3: Saving and Loading Infos
#
# This section demonstrates how to save and load Infos from a YAML file.

# %% [markdown]
# ### Save Infos to a YAML file

# %%
test_pth = Path(
    f"/tmp/test_safe_to_delete_{np.random.randint(low=1, high=2_000_000_000)}"
)
infos_save_fname = test_pth / "infos.yaml"
test_pth.mkdir(parents=True, exist_ok=True)

print(f"saving path: {infos_save_fname}")
infos.num_samples = {"train": 0}
infos.storage_backend = "zarr"
infos.save_to_file(infos_save_fname)

# %% [markdown]
# ### Load Infos from a YAML file

# %%
loaded_infos = Infos.from_path(infos_save_fname)
print(loaded_infos)

# %% [markdown]
# ### Load Infos from an explicit infos.yaml path

# %%
loaded_infos_from_explicit_path = Infos.from_path(test_pth / "infos.yaml")
print(loaded_infos_from_explicit_path)
