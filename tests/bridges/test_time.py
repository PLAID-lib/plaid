from time import time

import yaml, json
from pathlib import Path
from datasets import load_from_disk
import datasets
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from plaid import Dataset, Sample
from plaid.bridges import huggingface_bridge
from plaid.containers.features import SampleFeatures
from plaid.utils.base import get_mem
from plaid.utils.cgns_helper import (
    compare_cgns_trees_no_types,
    show_cgns_tree,
    flatten_cgns_tree,
    unflatten_cgns_tree
)
from plaid.utils.cgns_helper import flatten_cgns_tree, unflatten_cgns_tree


import numpy as np
import pyarrow as pa
import pickle

from Muscat.Bridges.CGNSBridge import CGNSToMesh


DATASET_NAME = "Tensile2d"
SPLIT_NAMES = ["train_500", "test"]


start = time()
repo_id = f"fabiencasenave/{DATASET_NAME}"
hf_dataset = huggingface_bridge.load_dataset_from_hub(repo_id, split = SPLIT_NAMES[0], num_proc = 4)
flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, "task_1")
infos = huggingface_bridge.load_infos_from_hub(repo_id)

cgns_types = key_mappings["cgns_types"]
end = time()
print("Time to instantiate new HF dataset from hub =", end - start)


# dataset_dict = huggingface_bridge.instantiate_plaid_datasetdict_from_hub(repo_id)
# sample = dataset_dict[SPLIT_NAMES[0]][0]
dataset = huggingface_bridge.to_plaid_dataset(hf_dataset, flat_cst, cgns_types)
# sample = dataset[0]
# tree = sample.features.data[0]
import copy
dataset[0].features.data[1.] = copy.deepcopy(dataset[0].features.data[0.])
dataset[0].features.data[2.] = copy.deepcopy(dataset[0].features.data[0.])
dataset[1].features.data[1.] = copy.deepcopy(dataset[1].features.data[0.])
dataset[1].features.data[3.] = copy.deepcopy(dataset[1].features.data[0.])

show_cgns_tree(dataset[0].features.data[2.])


# flat, cgns_types = flatten_cgns_tree(tree, 0)
# print(flat)
# print(cgns_types)

# unflat = unflatten_cgns_tree(flat, cgns_types, time=True)

# show_cgns_tree(unflat)
# print("---")
# show_cgns_tree(tree)

# print(compare_cgns_trees_no_types(unflat, tree))

main_splits = {"train_500":pb_def.get_split("train_500")}
hf_dataset_split, flat_cst, key_mappings = huggingface_bridge.plaid_dataset_to_huggingface_datasetdict(dataset, main_splits, verbose=True)

print(hf_dataset_split["train_500"].column_names)
print(hf_dataset_split["train_500"][0]["1/Global/p4"])
print(hf_dataset_split["train_500"][0]["0/Global/p4"])
# for i in range(500):
#     if hf_dataset_split["train_500"][0]["1/Global/p4"] is not None:
#         print(">>>>", i)

repo_out = f"fabiencasenave/{DATASET_NAME}_time"
huggingface_bridge.push_dataset_dict_to_hub(repo_out, hf_dataset_split)
huggingface_bridge.push_infos_to_hub(repo_out, infos)
huggingface_bridge.push_tree_struct_to_hub(repo_out, flat_cst, key_mappings)
huggingface_bridge.push_problem_definition_to_hub(repo_out, "task_1", pb_def)