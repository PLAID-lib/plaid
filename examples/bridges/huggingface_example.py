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


# %%
# Print Sample util
def show_sample(sample: Sample):
    print(f"sample = {sample}")
    sample.show_tree()
    print(f"{sample.get_scalar_names() = }")
    print(f"{sample.get_field_names() = }")


# %% [markdown]
# ## Initialize plaid dataset and problem_definition

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

print("Creating meshes dataset...")
for _ in range(3):
    mesh = MCT.CreateMeshOfTriangles(points, triangles)

    sample = Sample()

    sample.features.add_tree(MeshToCGNS(mesh))
    sample.add_scalar("scalar", np.random.randn())
    sample.add_field("node_field", np.random.rand(len(points)), location="Vertex")
    sample.add_field(
        "cell_field", np.random.rand(len(triangles)), location="CellCenter"
    )

    dataset.add_sample(sample)

infos = {
    "legal": {"owner": "Bob", "license": "my_license"},
    "data_production": {"type": "simulation", "physics": "3D example"},
}

dataset.set_infos(infos)

print(f" {dataset = }")

problem = ProblemDefinition()
problem.add_output_scalars_names(["scalar"])
problem.add_output_fields_names(["node_field", "cell_field"])
problem.add_input_meshes_names(["/Base/Zone"])

problem.set_task("regression")
problem.set_split({"train": [0, 1], "test": [2]})

print(f" {problem = }")

# %% [markdown]
# ## Section 1: Convert plaid dataset to Hugging Face
#
# The description field of Hugging Face dataset is automatically configured to include data from the plaid dataset info and problem_definition to prevent loss of information and equivalence of format.

# %%
hf_dataset = huggingface_bridge.plaid_dataset_to_huggingface(dataset, problem)
print()
print(f"{hf_dataset = }")
print(f"{hf_dataset.description = }")

# %% [markdown]
# The previous code generates a Hugging Face dataset containing all the samples from the plaid dataset, the splits being defined in the hf_dataset descriptions. For splits, Hugging Face proposes `DatasetDict`, which are dictionaries of hf datasets, with keys being the name of the corresponding splits. It is possible de generate a hf datasetdict directly from plaid:

# %%
hf_datasetdict = huggingface_bridge.plaid_dataset_to_huggingface_datasetdict(dataset, problem, main_splits = ['train', 'test'])
print()
print(f"{hf_datasetdict['train'] = }")
print(f"{hf_datasetdict['test'] = }")


# %% [markdown]
# ## Section 2: Generate a Hugging Face dataset with a generator

# %%
def generator():
    for id in range(len(dataset)):
        yield {
            "sample": pickle.dumps(dataset[id]),
        }


hf_dataset_gen = huggingface_bridge.plaid_generator_to_huggingface(
    generator, infos, problem
)
print()
print(f"{hf_dataset_gen = }")
print(f"{hf_dataset_gen.description = }")

# %% [markdown]
# The same is available with datasetdict:

# %%
hf_datasetdict_gen = huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
    generator, infos, problem, main_splits = ['train', 'test']
)
print()
print(f"{hf_datasetdict['train'] = }")
print(f"{hf_datasetdict['test'] = }")

# %% [markdown]
# ## Section 3: Convert a Hugging Face dataset to plaid
#
# Plaid dataset infos and problem_defitinion are recovered from the huggingface dataset

# %%
dataset_2, problem_2 = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset)
print()
print(f"{dataset_2 = }")
print(f"{dataset_2.get_infos() = }")
print(f"{problem_2 = }")

# %% [markdown]
# ## Section 4: Save and Load Hugging Face datasets
#
# ### From and to disk

# %%
# Save to disk
hf_dataset.save_to_disk("/tmp/path/to/dir")

# %%
# Load from disk
from datasets import load_from_disk

loaded_hf_dataset = load_from_disk("/tmp/path/to/dir")

print()
print(f"{loaded_hf_dataset = }")
print(f"{loaded_hf_dataset.description = }")

# %% [markdown]
# ### From and to the Hugging Face hub
#
# You need an huggingface account, with a configured access token, and to install huggingface_hub[cli].
# Pushing and loading a huggingface dataset without loss of information requires the configuration of a DatasetCard.
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
# Then, the following python instruction enable pushing a dataset to the hub:
# ```python
# hf_dataset.push_to_hub("chanel/dataset")
#
# from datasets import load_dataset_builder
#
# datasetInfo = load_dataset_builder("chanel/dataset").__getstate__()['info']
#
# from huggingface_hub import DatasetCard
#
# card_text = create_string_for_huggingface_dataset_card(
#     description = description,
#     download_size_bytes = datasetInfo.download_size,
#     dataset_size_bytes = datasetInfo.dataset_size,
#     ...)
# dataset_card = DatasetCard(card_text)
# dataset_card.push_to_hub("chanel/dataset")
# ```
#
# The second upload of the dataset_card is required to ensure that load_dataset from the hub will populate
# the hf-dataset.description field, and be compatible for conversion to plaid. Wihtout a dataset_card, the description field is lost.
#
#
# ### Load from hub
#
# #### General case
#
# ```python
# dataset = load_dataset("chanel/dataset", split="all_samples")
# ```
#
# More efficient retrieval are made possible by partial loads and  split loads (in the case of a datasetdict):
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
# To fully exploit optimzed data handling of the Hugging Face datasets library, it is possible to extract information from the huggingface dataset without converting to plaid. The ``description`` atttribute includes the plaid dataset _infos attribute and plaid problem_definition attributes.

# %%
print(f"{loaded_hf_dataset.description = }")

# %% [markdown]
# Get the first sample of the first split

# %%
split_names = list(loaded_hf_dataset.description["split"].keys())
id = loaded_hf_dataset.description["split"][split_names[0]]
hf_sample = loaded_hf_dataset[id[0]]

print(f"{hf_sample = }")

# %% [markdown]
# We notice that ``hf_sample`` is a binary object efficiently handled by huggingface datasets. It can be converted into a plaid sample using a specific constructor relying on a pydantic validator.

# %%
plaid_sample = huggingface_bridge.to_plaid_sample(hf_sample)

show_sample(plaid_sample)

# %% [markdown]
# Very large datasets can be streamed directly from the Hugging Face hub:
#
# ```python
# hf_dataset_stream = load_dataset("chanel/dataset", split="all_samples", streaming=True)
#
# plaid_sample = huggingface_bridge.to_plaid_sample(next(iter(hf_dataset_stream)))
#
# show_sample(plaid_sample)
# ```
#
# Or initialize a plaid dataset and problem definition for any number of samples relying on this streaming mechanisme:
#
# ```python
# from plaid.bridges.huggingface_bridge import streamed_huggingface_dataset_to_plaid
#
# dataset, pb_def = streamed_huggingface_dataset_to_plaid('PLAID-datasets/VKI-LS59', 2)
# ```



