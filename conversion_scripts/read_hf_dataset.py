"""Conversion scripts for reading PLAID-datasets on Hugging Face."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from time import time

import CGNS.PAT.cgnsutils as CGU

from plaid import Dataset, Sample
from plaid.bridges import huggingface_bridge
from plaid.utils.base import get_mem
from plaid.utils.cgns_helper import (
    compare_cgns_trees_no_types,
    flatten_cgns_tree,
    show_cgns_tree,
    unflatten_cgns_tree,
)
from tqdm import tqdm

repo_id = "fabiencasenave/Tensile2d"
split_names = ["train_500", "test", "OOD"]

repo_id = "fabiencasenave/2D_ElastoPlastoDynamics"
split_names = ["train", "test"]



# init_ram = get_mem()
# start = time()
# sample = huggingface_bridge.to_plaid_sample(
#     hf_dataset_new[split_names[0]], 0, flat_cst[split_names[0]], cgns_types
# )
# elapsed = time() - start
# print(
#     f"Time to build first sample of split {split_names[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
# )


hf_dataset_new = huggingface_bridge.load_dataset_from_hub(repo_id)
flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, "task_1")
infos = huggingface_bridge.load_infos_from_hub(repo_id)
cgns_types = key_mappings["cgns_types"]





init_ram = get_mem()
start = time()
for i in tqdm(range(len(hf_dataset_new["train"])), desc="Retrieving all variable features"):
    for path in key_mappings["variable_features"]:
        hf_dataset_new["train"].data[path][i].values.to_numpy(zero_copy_only=False)
elapsed = time() - start
print(
    f"Time to retrieve all data on train: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)


init_ram = get_mem()
start = time()
for i in tqdm(range(len(hf_dataset_new["train"])), desc="Retrieving all variable features"):
    sample = huggingface_bridge.to_plaid_sample(hf_dataset_new["train"], i, flat_cst["train"], cgns_types, enforce_shapes=False)
    for t in sample.get_all_mesh_times():
        for path in key_mappings["variable_features"]:
            sample.get_feature_by_path(path=path, time=t)
    del sample
elapsed = time() - start
print(
    f"Time to retrieve all data on train: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)




# show_cgns_tree(tree)