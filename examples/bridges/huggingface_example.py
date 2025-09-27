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

import numpy as np
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid.bridges import huggingface_bridge
from plaid import Dataset
from plaid import Sample
from plaid import ProblemDefinition
from plaid.types import FeatureIdentifier


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
# ## Section 1: Convert plaid dataset to Hugging Face Dataset

# %%
hf_dataset = huggingface_bridge.plaid_dataset_to_huggingface(dataset)
print()
print(f"{hf_dataset = }")

# %% [markdown]
# By default, all the indices from all splits are taken into account. One can generate a Hugging Face dataset for a given split by providing the problem_definition:

# %%
hf_dataset = huggingface_bridge.plaid_dataset_to_huggingface(dataset, pb_def, split="train")
print(hf_dataset)

# %% [markdown]
# The previous code generates a Hugging Face dataset containing all the samples from the plaid dataset, the splits being defined in the hf_dataset descriptions. For splits, Hugging Face proposes `DatasetDict`, which are dictionaries of hf datasets, with keys being the name of the corresponding splits. It is possible to generate a hf datasetdict directly from plaid:

# %%
hf_datasetdict = huggingface_bridge.plaid_dataset_to_huggingface_datasetdict(dataset, problem_definition = pb_def, main_splits = ['train', 'test'])
print()
print(f"{hf_datasetdict = }")


# %% [markdown]
# ## Section 2: Generate a Hugging Face dataset with a generator

# %%
def generator():
    for id in range(len(dataset)):
        yield {
            "sample": pickle.dumps(dataset[id]),
        }


hf_dataset_gen = huggingface_bridge.plaid_generator_to_huggingface(
    generator
)
print(f"{hf_dataset_gen = }")

# %% [markdown]
# The same is available with datasetdict:

# %%
hf_datasetdict = huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
    generator, main_splits = ['train', 'test']
)
print(f"{hf_datasetdict = }")

# %% [markdown]
# ## Section 3: Convert a Hugging Face dataset to plaid

# %%
dataset_2 = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset)
print()
print(f"{dataset_2 = }")

# %% [markdown]
# ## Section 4: Save and Load Hugging Face datasets
#
# ### From and to disk
#
# Saving datasetdict, infos and problem definition to disk:

# %%
huggingface_bridge.save_dataset_dict_to_disk("/tmp/test_dir", hf_datasetdict)
huggingface_bridge.save_dataset_infos_to_disk("/tmp/test_dir", infos)
huggingface_bridge.save_problem_definition_to_disk("/tmp/test_dir", "task_1", pb_def)

# %% [markdown]
# Loading datasetdict, infos and problem definition from disk:

# %%
loaded_hf_datasetdict = huggingface_bridge.load_dataset_dict_from_to_disk("/tmp/test_dir")
loaded_infos = huggingface_bridge.load_dataset_infos_from_disk("/tmp/test_dir")
loaded_pb_def = huggingface_bridge.load_problem_definition_from_disk("/tmp/test_dir", "task_1")

print(f"{loaded_hf_datasetdict = }")
print(f"{loaded_infos = }")
print(f"{loaded_pb_def = }")

# %% [markdown]
# ### From and to the Hugging Face hub
#
# To save a dataset on the Hub, you need an huggingface account, with a configured access token, and to install huggingface_hub[cli].
#
# Find below example of instruction (not executed by this notebook).
#
# ### Push to the hub
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
# huggingface_bridge.push_dataset_infos_to_hub("chanel/dataset", infos)
# huggingface_bridge.push_problem_definition_to_hub("chanel/dataset", pb_def, "location")
# ```
#
# The dataset card can then be customized online, on the dataset repo page directly.

# %% [markdown]
# ### Load from hub
#
# #### General case
#
# Retrieval are made possible by partial loads and split loads:
#
# ```python
# dataset_train = load_dataset("chanel/dataset", split="train")
# dataset_train_extract = load_dataset("chanel/dataset", split="train[:10]")
# ```
#
# #### Proxy
#
# A retrieval function robust to cases where you are behind a proxy and relying on a private mirror is avalable;
#
# ```python
# from plaid.bridges.huggingface_bridge import load_hf_dataset_from_hub
# hf_dataset = load_hf_dataset_from_hub("chanel/dataset", *args, **kwargs)
# ```
#
# - Streaming mode is not supported when using a private mirror.
# - Falls back to local download if streaming or public loading fails.
# - To use behind a proxy, you may need to set:
#   - `HF_ENDPOINT` to your private mirror address
#   - `CURL_CA_BUNDLE` to your trusted CA certificates
#   - `HF_HOME` to a shared cache directory if needed

# %% [markdown]
# ## Section 5: Handle plaid samples from Hugging Face datasets without converting the complete dataset to plaid
#
# To fully exploit optimzed data handling of the Hugging Face datasets library, it is possible to extract information from the huggingface dataset without converting to plaid.


# %% [markdown]
# Get the first sample of the first split

# %%
hf_sample = hf_dataset[0]

print(f"{hf_sample = }")

# %% [markdown]
# We notice that ``hf_sample`` contains a binary object efficiently handled by huggingface datasets. It can be converted into a plaid sample using a specific constructor relying on a pydantic validator.

# %%
plaid_sample = huggingface_bridge.to_plaid_sample(hf_sample)

show_sample(plaid_sample)


# %% [markdown]
# Very large datasets can be streamed directly from the Hugging Face hub:
#
# ```python
# hf_dataset_stream = load_dataset("chanel/dataset", split="train", streaming=True)
# plaid_sample = huggingface_bridge.to_plaid_sample(next(iter(hf_dataset_stream)))
# ```
#
# If you are behing a proxy:
# ```python
# hf_dataset_stream = huggingface_bridge.load_hf_dataset_from_hub("chanel/dataset", split="train", streaming=True)
# plaid_sample = huggingface_bridge.to_plaid_sample(next(iter(hf_dataset_stream)))
# ```
