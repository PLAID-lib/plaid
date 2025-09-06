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
# # Dataset Examples
#
# This Jupyter Notebook demonstrates various use cases for the Dataset class, including:
#
# 1. Initializing an Empty Dataset and Adding Samples
# 2. Retrieving and Manipulating Samples from a Dataset
# 3. Performing Operations on the Dataset
# 4. Saving and Loading Datasets from directories or files
#
# This notebook provides detailed examples of using the Dataset class to manage data, Samples, and information within a PLAID Dataset. It is intended for documentation purposes and familiarization with the PLAID library.
#
# **Each section is documented and explained.**

# %%
# Import required libraries
from pathlib import Path
import platform

import numpy as np

# %%
# Import necessary libraries and functions
import Muscat.MeshContainers.ElementsDescription as ElementsDescription
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

import plaid
from plaid import Dataset
### The 2 following commented lines are equivalent to the one above
# from plaid.containers import Dataset
# from plaid import Dataset
from plaid import Sample


# %%
# Print dict util
def dprint(name: str, dictio: dict, end: str = "\n"):
    print(name, "{")
    for key, value in dictio.items():
        print("    ", key, ":", value)

    print("}", end=end)


# %% [markdown]
# ## Section 1: Initializing an Empty Dataset and Samples construction
#
# This section demonstrates how to initialize an empty Dataset and handle Samples.

# %% [markdown]
# ### Initialize an empty Dataset

# %%
print("#---# Empty Dataset")
dataset = Dataset()
print(f"{dataset=}")

# %% [markdown]
# ### Create Sample

# %%
# Create Sample
points = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.5],
    ]
)

triangles = np.array(
    [
        [0, 1, 2],
        [0, 2, 3],
        [2, 4, 3],
    ]
)

bars = np.array([[0, 1], [0, 2]])

Mesh = MCT.CreateMeshOfTriangles(points, triangles)
elbars = Mesh.GetElementsOfType(ElementsDescription.Bar_2)
elbars.AddNewElements(bars, [1, 2])
cgns_mesh = MeshToCGNS(Mesh)

# Initialize an empty Sample
print("#---# Empty Sample")
sample_01 = Sample()
print(f"{sample_01 = }")

# %%
# Add a CGNS tree structure to the Sample
sample_01.add_tree(cgns_mesh)
print(f"{sample_01 = }")

# %%
# Add a scalar to the Sample
sample_01.add_scalar("rotation", np.random.randn())
print(f"{sample_01 = }")

# %% [markdown]
# ### Print Sample general data

# %%
# Initialize another empty Sample
print("#---# Empty Sample")
sample_02 = Sample()
print(f"{sample_02 = }")

# %%
# Add a scalar to the second Sample
sample_02.add_scalar("rotation", np.random.randn())
print(f"{sample_02 = }")

# %% [markdown]
# ### Display Sample CGNS tree

# %%
# Initialize a third empty Sample
print("#---# Empty Sample")
sample_03 = Sample()
sample_03.add_scalar("speed", np.random.randn())
sample_03.add_scalar("rotation", sample_01.get_scalar("rotation"))
sample_03.add_tree(cgns_mesh)

# Show Sample CGNS content
sample_03.show_tree()

# %%
# Add a field to the third empty Sample
sample_03.add_field("temperature", np.random.rand(5), "Zone", "Base_2_2")
sample_03.show_tree()

# %% [markdown]
# ### Get Sample data

# %%
# Print sample general data
print(f"{sample_03 = }", end="\n\n")

# Print sample scalar data
print(f"{sample_03.get_scalar_names() = }")
print(f"{sample_03.get_scalar('speed') = }")
print(f"{sample_03.get_scalar('rotation') = }", end="\n\n")

# Print sample scalar data
print(f"{sample_03.get_field_names() = }")
print(f"{sample_03.get_field('temperature') = }")

# %% [markdown]
# ## Section 2: Performing Operations on the Dataset
#
# This section demonstrates how to add Samples to the Dataset, add information, and access data.

# %% [markdown]
# ### Add Samples in the Dataset

# %%
# Add Samples by id in the Dataset
dataset.set_sample(id=0, sample=sample_01)
dataset.set_sample(1, sample_02)

# Add unique Sample and automatically create its id
added_sample_id = dataset.add_sample(sample_03)
print(f"{added_sample_id = }")

# %% [markdown]
# ### Iterate through Samples in the Dataset

# %%
for sample in dataset:
    print(sample)

# %% [markdown]
# ### Add and display information to the Dataset

# %%
# Add node information to the Dataset
dataset.add_info("legal", "owner", "Safran")

# Retrive dataset information
import json

dataset_info = dataset.get_infos()
print("dataset info =", json.dumps(dataset_info, sort_keys=False, indent=4), end="\n\n")

# Overwrite information (logger will display warnings)
infos = {"legal": {"owner": "Safran", "license": "CC0"}}
dataset.set_infos(infos)

# Retrive dataset information
dataset_info = dataset.get_infos()
print("dataset info =", json.dumps(dataset_info, sort_keys=False, indent=4), end="\n\n")

# Add tree information to the Dataset (logger will display warnings)
dataset.add_infos("data_description", {"number_of_samples": 0, "number_of_splits": 0})

# Pretty print dataset information
dataset.print_infos()

# %% [markdown]
# ### Get a list of specific Samples in a Dataset

# %%
get_samples_from_ids = dataset.get_samples(ids=[0, 1])
dprint("get samples from ids =", get_samples_from_ids)

# %% [markdown]
# ### Get the list of Sample ids in a Dataset

# %%
# Print sample IDs
print("get_sample_ids =", dataset.get_sample_ids())

# %% [markdown]
# ### Print Dataset general data

# %%
# Print the Dataset
print(f"{dataset = }")
print("length of dataset =", len(dataset))

# %% [markdown]
# ### Add a list of Sample to a Dataset

# %%
# Create a new Dataset and add multiple samples
dataset = Dataset()
samples = [sample_01, sample_02, sample_03]
added_ids = dataset.add_samples(samples)
print(f"{added_ids = }")
print(f"{dataset = }")

# %% [markdown]
# ### Access to Samples data through Dataset

# %%
# Access Sample data with indexes through the Dataset
print(f"{dataset(0) = }")  # call strategy
print(f"{dataset[1] = }")  # getitem strategy
print(f"{dataset[2] = }", end="\n\n")

print("scalar of the first sample = ", dataset[0].get_scalar_names())
print("scalar of the second sample = ", dataset[1].get_scalar_names())
print("scalar of the third sample = ", dataset[2].get_scalar_names())

# %%
# Access dataset information
print(f"{dataset[0].get_scalar('rotation') = }")
print(f"{dataset[1].get_scalar('rotation') = }")
print(f"{dataset[2].get_scalar('rotation') = }")

# %% [markdown]
# ### Get Dataset scalars to tabular

# %%
# Print scalars in tabular format
print(f"{dataset.get_scalar_names() = }", end="\n\n")

dprint("get rotation scalar = ", dataset.get_scalars_to_tabular(["rotation"]))
dprint("get speed scalar = ", dataset.get_scalars_to_tabular(["speed"]), end="\n\n")

# Get specific scalars in tabular format
dprint("get specific scalars =", dataset.get_scalars_to_tabular(["speed", "rotation"]))
dprint("get all scalars =", dataset.get_scalars_to_tabular())

# %%
# Get specific scalars np.array
print("get all scalar arrays = ", dataset.get_scalars_to_tabular(as_nparray=True))

# %% [markdown]
# ### Get Dataset fields

# %%
# Print fields in the Dataset
print("fields in the dataset = ", dataset.get_field_names())

# %% [markdown]
# ## Section 3: Various operations on the Dataset
#
# This section demonstrates operations like merging datasets, adding tabular scalars, and setting information.

# %% [markdown]
# ### Initialize a Dataset with a list of Samples

# %%
# Create another Dataset
other_dataset = Dataset()
nb_samples = 3
samples = []
for _ in range(nb_samples):
    sample = Sample()
    sample.add_scalar("rotation", np.random.rand() + 1.0)
    sample.add_scalar("random_name", np.random.rand() - 1.0)
    samples.append(sample)

# Add a list of Samples
other_dataset.add_samples(samples)
print(f"{other_dataset = }")

# %% [markdown]
# ### Merge two Datasets

# %%
# Merge the other dataset with the main dataset
print(f"before merge: {dataset = }")
dataset.merge_dataset(other_dataset)
print(f"after merge: {dataset = }", end="\n\n")

dprint("dataset scalars = ", dataset.get_scalars_to_tabular())

# %% [markdown]
# ### Add tabular scalars to a Dataset

# %%
# Adding tabular scalars to the dataset
new_scalars = np.random.rand(3, 2)
dataset.add_tabular_scalars(new_scalars, names=["Tu", "random_name"])

print(f"{dataset = }")
dprint("dataset scalars =", dataset.get_scalars_to_tabular())

# %% [markdown]
# ### Set additional information to a dataset

# %%
infos = {
    "legal": {"owner": "Safran", "license": "CC0"},
    "data_production": {"type": "simulation", "simulator": "dummy"},
}
dataset.set_infos(infos)
dataset.print_infos()

# %% [markdown]
# ## Section 4: Saving and Loading Dataset
#
# This section demonstrates how to save and load a Dataset from a directory or file.

# %% [markdown]
# ### Save a Dataset as a file tree

# %%
tmpdir = f"/tmp/test_safe_to_delete_{np.random.randint(low=1, high=2_000_000_000)}"
print(f"Save dataset in: {tmpdir}")

dataset._save_to_dir_(tmpdir)

# %% [markdown]
# ### Get the number of Samples that can be loaded from a directory

# %%
nb_samples = plaid.get_number_of_samples(tmpdir)
print(f"{nb_samples = }")

# %% [markdown]
# ### Load a Dataset from a directory via initialization

# %%
loaded_dataset_from_init = Dataset(tmpdir)
print(f"{loaded_dataset_from_init = }")

if platform.system() == "Linux":
    multi_process_loaded_dataset = Dataset(tmpdir, processes_number=3)
    print(f"{multi_process_loaded_dataset = }")

# %% [markdown]
# ### Load a Dataset from a directory via the Dataset class

# %%
loaded_dataset_from_class = Dataset.load_from_dir(tmpdir)
print(f"{loaded_dataset_from_class = }")

if platform.system() == "Linux":
    multi_process_loaded_dataset = Dataset.load_from_dir(tmpdir, processes_number=3)
    print(f"{multi_process_loaded_dataset = }")

# %% [markdown]
# ### Load the dataset from a directory via a Dataset instance

# %%
loaded_dataset_from_instance = Dataset()
loaded_dataset_from_instance._load_from_dir_(tmpdir)

print(f"{loaded_dataset_from_instance = }")

if platform.system() == "Linux":
    multi_process_loaded_dataset = Dataset()
    multi_process_loaded_dataset._load_from_dir_(tmpdir, processes_number=3)
    print(f"{multi_process_loaded_dataset = }")

# %% [markdown]
# ### Save the dataset to a TAR (Tape Archive) file

# %%
tmpdir = Path(f"/tmp/test_safe_to_delete_{np.random.randint(low=1, high=2_000_000_000)}")
tmpfile = tmpdir / "test_file.plaid"

print(f"Save dataset in: {tmpfile}")
dataset.save(tmpfile)

# %% [markdown]
# ### Load the dataset from a TAR (Tape Archive) file via Dataset instance

# %%
new_dataset = Dataset()
new_dataset.load(tmpfile)

print(f"{dataset = }")
print(f"{new_dataset = }")

# %% [markdown]
# ### Load the dataset from a TAR (Tape Archive) file via initialization

# %%
new_dataset = Dataset(tmpfile)

print(f"{dataset = }")
print(f"{new_dataset = }")
