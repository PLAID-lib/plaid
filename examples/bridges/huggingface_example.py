# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: plaid-dev
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hugging Face support
#
# This Jupyter Notebook demonstrates various operations involving the Hugging Face bridge:
#
# 1. Converting a plaid dataset to Hugging Face
# 2. Generating a Hugging Face dataset with a generator
# 3. Converting a Hugging Face dataset to plaid
# 4. Saving and Loading Hugging Face datasets
# 5. Handling plaid samples from Hugging Face datasets without converting the complete dataset to plaid
#
#
# **Each section is documented and explained.**

# %%
# Import necessary libraries and functions
import pickle
import tempfile
import shutil
from time import time

import numpy as np
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid.bridges import huggingface_bridge
from plaid import Dataset, Sample, ProblemDefinition
from plaid.types import FeatureIdentifier
from plaid.utils.base import get_mem


# %%
# Print Sample util
def show_sample(sample: Sample):
    print(f"sample = {sample}")
    sample.show_tree()
    print(f"{sample.get_scalar_names() = }")
    print(f"{sample.get_field_names() = }")


# %% [markdown]
# ## Initialize plaid dataset, infos and problem_definition

# %%
# Input data
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

dataset = Dataset()

scalar_feat_id = FeatureIdentifier({"type": "scalar", "name": "scalar"})
node_field_feat_id = FeatureIdentifier({"type": "field", "name": "node_field", "location": "Vertex"})
cell_field_feat_id = FeatureIdentifier({"type": "field", "name": "cell_field", "location": "CellCenter"})

print("Creating meshes dataset...")
for _ in range(3):
    mesh = MCT.CreateMeshOfTriangles(points, triangles)

    sample = Sample()

    sample.add_tree(MeshToCGNS(mesh, exportOriginalIDs = False))

    sample.update_features_from_identifier(scalar_feat_id, np.random.randn(), in_place=True)
    sample.update_features_from_identifier(node_field_feat_id, np.random.rand(len(points)), in_place=True)
    sample.update_features_from_identifier(cell_field_feat_id, np.random.rand(len(triangles)), in_place=True)

    dataset.add_sample(sample)

infos = {
    "legal": {"owner": "Bob", "license": "my_license"},
    "data_production": {"type": "simulation", "physics": "3D example"},
}

dataset.set_infos(infos)

print(f" {dataset = }")
print(f" {infos = }")

pb_def = ProblemDefinition()
pb_def.add_in_features_identifiers([scalar_feat_id, node_field_feat_id])
pb_def.add_out_features_identifiers([cell_field_feat_id])

pb_def.set_task("regression")
pb_def.set_split({"train": [0, 1], "test": [2]})

print(f" {pb_def = }")

# %% [markdown]
# ## Section 1: Convert plaid datasets to Hugging Face DatasetDict

# %%
main_splits = {
    split_name: pb_def.get_split(split_name) for split_name in ["train", "test"]
}

hf_datasetdict, flat_cst, key_mappings = huggingface_bridge.plaid_dataset_to_huggingface_datasetdict(dataset, main_splits)

print(f"{hf_datasetdict = }")
print(f"{flat_cst = }")
print(f"{key_mappings = }")

# %% [markdown]
# A partitioning of all the indices is provided in `main_splits`. The conversion outputs `flat_cst` and `key_mappings`, which are central to the Hugging Face support:
# - **`flat_cst`**: constant features dictionary (path → value): a flatten tree containing the CGNS trees leaves that a reconstant throughout the plaid dataset.
# - **`key_mappings`**: metadata dictionary containing keys such as:
#     - `variable_features`: list of paths for non-constant features.
#     - `constant_features`: list of paths for constant features.
#     - `cgns_types`: mapping from paths to CGNS types.
#
# `flat_cst` and `cgns_types` are required for reconstructing plaid datasets and samples from the hugginface datasets.

# %% [markdown]
# ## Section 2: Generate a Hugging Face dataset with a generator

# %% [markdown]
# Ganarators are used to handle large datasets that do not fit in memory:

# %%
generators = {}
for split_name, ids in main_splits.items():
    def generator_(ids=ids):
        for id in ids:
            yield dataset[id]
    generators[split_name] = generator_

hf_datasetdict, flat_cst, key_mappings = (
    huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
        generators
    )
)
print(f"{hf_datasetdict = }")
print(f"{flat_cst = }")
print(f"{key_mappings = }")

# %% [markdown]
# In this example, the generators are not very usefull since the plaid dataset is already loaded in memory. In real settings, one can create generators in the following way to prevent loading all the data beforehand:
# ```python
# generators = {}
# for split_name, ids in main_splits.items():
#     def generator_(ids=ids):
#         for id in ids:
#             loaded_simulation_data = load('path/to/split_name/simulation_id')
#             sample = convert_to_sample(loaded_simulation_data)
#             yield sample
#     generators[split_name] = generator_
# ```

# %% [markdown]
# ## Section 3: Convert a Hugging Face dataset to plaid

# %%
cgns_types = key_mappings["cgns_types"]

dataset_2 = huggingface_bridge.to_plaid_dataset(hf_datasetdict['train'], flat_cst, cgns_types)
print()
print(f"{dataset_2 = }")

# %% [markdown]
# ## Section 4: Save and Load Hugging Face datasets
#
# ### From and to disk
#
# Saving and loading datasetdict, infos, tree_struct and problem definition to disk:

# %%
with tempfile.TemporaryDirectory() as out_dir:

    huggingface_bridge.save_dataset_dict_to_disk(out_dir, hf_datasetdict)
    huggingface_bridge.save_infos_to_disk(out_dir, infos)
    huggingface_bridge.save_tree_struct_to_disk(out_dir, flat_cst, key_mappings)
    huggingface_bridge.save_problem_definition_to_disk(out_dir, "task_1", pb_def)

    loaded_hf_datasetdict = huggingface_bridge.load_dataset_from_disk(out_dir)
    loaded_infos = huggingface_bridge.load_infos_from_disk(out_dir)
    flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_disk(out_dir)
    loaded_pb_def = huggingface_bridge.load_problem_definition_from_disk(out_dir, "task_1")

    shutil.rmtree(out_dir)

print(f"{loaded_hf_datasetdict = }")
print(f"{loaded_infos = }")
print(f"{flat_cst = }")
print(f"{key_mappings = }")
print(f"{loaded_pb_def = }")

# %% [markdown]
# ### From and to the Hugging Face hub
#
# Find below examples of instructions (not executed by this notebook).

# %% [markdown]
# #### Load from hub
#
# To load datasetdict, infos and problem_definitions from the hub:
# ```python
# huggingface_bridge.load_dataset_from_hub("chanel/dataset", *args, **kwargs)
# huggingface_bridge.load_hf_infos_from_hub("chanel/dataset")
# huggingface_bridge.load_hf_problem_definition_from_hub("chanel/dataset", "name")
# ```
#
# Partial retrieval are possible along samples
# ```python
# huggingface_bridge.load_dataset_from_hub("chanel/dataset", split="train[:10], *args, **kwargs)
# ```
#
# Streaming allows handling very large datasets
# ```python
# hf_dataset_streamed = huggingface_bridge.load_dataset_from_hub("chanel/dataset", split="split", streaming=True, *args, **kwargs)
# for hf_sample in hf_dataset_streamed:
#     sample = huggingface_bridge.to_plaid_sample(hf_sample, flat_cst, cgns_types)
# ```
#
# Native HF datasets commands are also possible:
#
# ```python
# dataset_train = load_dataset("chanel/dataset", split="train")
# dataset_train = load_dataset("chanel/dataset", split="train", streaming=True)
# dataset_train_extract = load_dataset("chanel/dataset", split="train[:10]")
# ```
#
# If you are behind a proxy and relying on a private mirror the function `load_dataset_from_hub` is working provided the following is set:
# - `HF_ENDPOINT` to your private mirror address
# - `CURL_CA_BUNDLE` to your trusted CA certificates
# - `HF_HOME` to a shared cache directory if needed

# %% [markdown]
# #### Push to the hub
#
# To push a dataset on the Hub, you need an huggingface account, with a configured access token.
#
# First login the huggingface cli:
# ```bash
# huggingface-cli login
# ```
# and enter you access token.
#
# Then, the following python instruction enable pushing datasetdict, infos and problem_definitions to the hub:
# ```python
# huggingface_bridge.push_dataset_dict_to_hub("chanel/dataset", hf_dataset_dict)
# huggingface_bridge.push_infos_to_hub("chanel/dataset", infos)
# huggingface_bridge.push_tree_struct_to_hub("chanel/dataset", flat_cst, key_mappings)
# huggingface_bridge.push_problem_definition_to_hub("chanel/dataset", "location", pb_def)
# ```
#
# The dataset card can then be customized online, on the dataset repo page directly.

# %% [markdown]
# ## Section 5: Handle plaid samples from Hugging Face datasets without converting the complete dataset to plaid
#
# To fully exploit optimzed data handling of the Hugging Face datasets library, it is possible to extract information from the huggingface dataset without converting to plaid.


# %% [markdown]
# Get the first sample of the first split

# %%
hf_sample = hf_datasetdict['train'][0]

print(f"{hf_sample = }")

# %% [markdown]
# We notice that ``hf_sample`` is not a plaid sample, but a dict containing the variable features of the datasets, with keys being the flattened path of the CGNS tree. contains a binary object efficiently handled by huggingface datasets. It can be converted into a plaid sample using a specific constructor relying on a pydantic validator, and the required `flat_cst` and `cgns_types`.

# %%
plaid_sample = huggingface_bridge.to_plaid_sample(hf_sample, flat_cst, cgns_types)

show_sample(plaid_sample)


# %% [markdown]
# Very large datasets that do not fit on disk can be streamed directly from the Hugging Face hub:
#
# ```python
# hf_dataset_stream = load_dataset("chanel/dataset", split="train", streaming=True)
# plaid_sample = huggingface_bridge.to_plaid_sample(next(iter(hf_dataset_stream)), flat_cst, cgns_types)
# ```
#
# If you are behing a proxy:
# ```python
# hf_dataset_stream = huggingface_bridge.load_dataset_from_hub("chanel/dataset", split="train", streaming=True)
# plaid_sample = huggingface_bridge.to_plaid_sample(next(iter(hf_dataset_stream)), flat_cst, cgns_types)
# ```

# %% [markdown]
# ## Section 6: Advanced concepts

# %% [markdown]
# In this section, we investigate concepts to better exploit the datasets made available on Hugging Face, by looking into read speed and memory usage. The commands are not executed by this notebook. You can copy/paste the following code to execute it, but be mindfull that it will download a 235MB dataset.
#
# ```python
# repo_id = "fabiencasenave/Tensile2d_DO_NOT_DELETE"
# split_names = ["train_500", "test", "OOD"]
#
# hf_dataset_dict = huggingface_bridge.load_dataset_from_hub(repo_id)
# ```

# %% [markdown]
# We investigate the time and memory needed to instantiate the plaid dataset dict from the repo_id, now that the hf datasets have been loaded in cache:
# ```python
# init_ram = get_mem()
# start = time()
# dataset_dict = huggingface_bridge.instantiate_plaid_datasetdict_from_hub(repo_id)
# elapsed = time() - start
# print(f"Time to instantiate plaid dataset dict from cache: {elapsed:.6g} s, RAM usage increase: {get_mem()-init_ram} MB")
# ```
# ```bash
# >> Time to instantiate plaid dataset dict from cache: 1.37948 s, RAM usage increase: 22.5 MB
# ```
# We notice the RAM usage is lower than the size of the dataset: all the variable shape 1DArrays and constant shape 2DArrays in the samples are initiated in no-copy mode.

# %% [markdown]
# We now investigate the possible gains when handling the datasets directly. First, bypassing cache checks and constructing plaid dataset from an instantiated HF dataset is much faster:
# ```python
# flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
# pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, "task_1")
# infos = huggingface_bridge.load_infos_from_hub(repo_id)
# cgns_types = key_mappings["cgns_types"]
#
# hf_dataset = hf_dataset_dict[split_names[0]]
#
# init_ram = get_mem()
# start = time()
# dataset = huggingface_bridge.to_plaid_dataset(hf_dataset, flat_cst, cgns_types)
# elapsed = time() - start
# print(f"Time to build dataset on split {split_names[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem()-init_ram} MB")
# ```
# ```bash
# >> Time to build dataset on split train_500: 0.173115 s, RAM usage increase: 16.3125 MB
# ```

# %% [markdown]
# It is possible to further remove overheads by accessing directly 1DArrays in the arrow table of the HF datasets in no-copy mode:
# ```python
# init_ram = get_mem()
# start = time()
# data = {}
# for i in range(len(hf_dataset)):
#     data[i] = hf_dataset.data["Base_2_2/Zone/PointData/sig12"][i].values.to_numpy(zero_copy_only=True)
# elapsed = time() - start
# print(f"Time to read 1D fields of variable size on the complete split {split_names[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem()-init_ram} MB")
# ```
# ```bash
# >> Time to read 1D fields of variable size on the complete split train_500: 0.0021801 s, RAM usage increase: 0.375 MB
# ```
