"""Conversion scripts for reading PLAID-datasets on Hugging Face."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import os
from pathlib import Path
from time import time

import psutil
import yaml
from tqdm import tqdm

from plaid.bridges import huggingface_bridge


def get_mem():
    """Get the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # in MB


# dataset_name = "Tensile2d"
dataset_name = "VKI-LS59"

# repo_id = "fabiencasenave/VKI-LS59"
# split_names = ["train", "test"]

# repo_id = "fabiencasenave/2D_profile"
# split_names = ["train", "test"]

# repo_id = "fabiencasenave/Rotor37"
# split_names = ["train_1000", "test"]

# repo_id = "fabiencasenave/2D_Multiscale_Hyperelasticity"
# split_names = ["DOE_train", "DOE_test"]

# repo_id = "fabiencasenave/2D_ElastoPlastoDynamics"
# split_names = ["train", "test"]


with Path(f"./config_{dataset_name}.yaml").open("r") as file:
    data_config = yaml.safe_load(file)

split_names = data_config["split_names_out"]
pb_def_names = data_config["pb_def_names"]
repo_id = data_config["repo_id_out"]


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
pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, pb_def_names[0])
infos = huggingface_bridge.load_infos_from_hub(repo_id)
cgns_types = key_mappings["cgns_types"]


# local_repo = "Tensile2d"
# hf_dataset_new = huggingface_bridge.load_dataset_from_disk(local_repo)
# flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_disk(local_repo)
# pb_def = huggingface_bridge.load_problem_definition_from_disk(
#     local_repo, pb_def_names[0]
# )
# infos = huggingface_bridge.load_infos_from_disk(local_repo)
# cgns_types = key_mappings["cgns_types"]


init_ram = get_mem()
start = time()
for i in tqdm(
    range(len(hf_dataset_new[split_names[0]])), desc="Retrieving all variable features"
):
    for path in pb_def.get_out_features_identifiers():
        hf_dataset_new[split_names[0]].data[path][i].values.to_numpy(
            zero_copy_only=False
        )
elapsed = time() - start
print(
    f"Time to retrieve out features on train: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)


init_ram = get_mem()
start = time()
for i in tqdm(
    range(len(hf_dataset_new[split_names[0]])), desc="Retrieving all variable features"
):
    sample = huggingface_bridge.to_plaid_sample(
        hf_dataset_new[split_names[0]],
        i,
        flat_cst[split_names[0]],
        cgns_types,
        enforce_shapes=False,
    )
    for t in sample.get_all_mesh_times():
        for path in pb_def.get_in_features_identifiers():
            sample.get_feature_by_path(path=path, time=t)
        for path in pb_def.get_out_features_identifiers():
            sample.get_feature_by_path(path=path, time=t)
    del sample
elapsed = time() - start
print(
    f"Time to retrieve in and out features on train: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)
