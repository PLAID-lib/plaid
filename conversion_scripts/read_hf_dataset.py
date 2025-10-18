"""Conversion scripts for reading PLAID-datasets on Hugging Face."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import os
from time import time

import psutil
from tqdm import tqdm

from plaid.bridges import huggingface_bridge

repo_id = "fabiencasenave/Tensile2d"
split_names = ["train_500", "test", "OOD"]
pb_def_names = [
    "regression_8",
    "regression_16",
    "regression_32",
    "regression_64",
    "regression_125",
    "regression_250",
    "regression_500",
]

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


def get_mem():
    """Get the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # in MB


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
