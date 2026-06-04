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
# metadata, access entries with mapping-style helpers, and save/load infos.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
from pathlib import Path

import numpy as np

# %%
# Import necessary libraries and classes
from plaid.infos import DataProduction, Infos, Legal

# %% [markdown]
# ## Section 1: Initializing Infos
#
# This section demonstrates how to initialize Infos with the current API.

# %% [markdown]
# ### Initialize and print Infos

# %%
print("#---# Infos")
infos = Infos(
    legal=Legal(owner="PLAID", license="MIT"),
    num_samples={"train": 2, "test": 2},
    storage_backend="cgns",
)
print(f"{infos = }")

# %% [markdown]
# ### Initialize Infos from a plain mapping

# %%
infos_from_mapping = Infos.from_mapping(
    {
        "legal": {
            "owner": "PLAID",
            "license": "MIT",
        },
        "data_description": "Example metadata for a PLAID dataset.",
        "num_samples": {"train": 2, "test": 2},
        "storage_backend": "cgns",
    }
)
print(f"{infos_from_mapping = }")

# %% [markdown]
# ## Section 2: Configuring Infos and retrieve data
#
# This section demonstrates how to handle and configure Infos objects and access
# metadata.

# %% [markdown]
# ### Set legal metadata

# %%
infos.legal = Legal(owner="Safran", license="proprietary")
print(f"{infos.legal = }")

# %% [markdown]
# ### Set data production metadata

# %%
infos.data_production = DataProduction(
    type="simulation",
    physics="fluid dynamics",
    simulator="ExampleSolver",
    hardware="ExampleCluster",
    computation_duration="1 hour",
    script="run_simulation.py",
    contact="contact@example.com",
)
print(f"{infos.data_production = }")

# %% [markdown]
# ### Set data description, sample counts, and storage backend

# %%
infos.data_description = "Example dataset generated for the Infos example."
infos.num_samples = {"train": 3, "test": 1}
infos.storage_backend = "zarr"

print(f"{infos.data_description = }")
print(f"{infos.num_samples = }")
print(f"{infos.storage_backend = }")

# %% [markdown]
# ### Retrieve data with mapping-style helpers

# %%
print(f"{infos['legal'] = }")
print(f"{infos.get('storage_backend') = }")
print(f"{infos.to_dict() = }")

# %% [markdown]
# ## Section 3: Saving and Loading Infos
#
# This section demonstrates how to save and load Infos from a directory or YAML
# file.

# %% [markdown]
# ### Save Infos to a YAML file

# %%
test_pth = Path(
    f"/tmp/test_safe_to_delete_{np.random.randint(low=1, high=2_000_000_000)}"
)
infos_save_fname = test_pth / "infos.yaml"
test_pth.mkdir(parents=True, exist_ok=True)
print(f"saving path: {infos_save_fname}")

infos.save_to_file(infos_save_fname)

# %% [markdown]
# ### Load Infos from a YAML file

# %%
loaded_infos = Infos.from_path(infos_save_fname)
print(loaded_infos)

# %% [markdown]
# ### Load Infos from a directory containing infos.yaml

# %%
loaded_infos_from_dir = Infos.from_path(test_pth)
print(loaded_infos_from_dir)