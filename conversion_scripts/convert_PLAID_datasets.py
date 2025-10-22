"""Conversion scripts for binary to arrow native types for PLAID-datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import os
from time import time

import psutil

from plaid import Dataset, ProblemDefinition, Sample
from plaid.bridges import huggingface_bridge
from plaid.utils.cgns_helper import (
    compare_cgns_trees_no_types,
    flatten_cgns_tree,
    show_cgns_tree,
    unflatten_cgns_tree,
)


def get_mem():
    """Get the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)  # in MB


# ------------------------------------------
# Choose repo to convert
# ------------------------------------------

repo_id = "PLAID-datasets/Tensile2d"
split_names = ["train_500", "test", "OOD"]
split_names_out = ["train", "test", "OOD"]
pb_def_names = [
    "regression_8",
    "regression_16",
    "regression_32",
    "regression_64",
    "regression_125",
    "regression_250",
    "regression_500",
    "PLAID_benchmark",
]
train_split_names = [
    "train_8",
    "train_16",
    "train_32",
    "train_64",
    "train_125",
    "train_250",
    "train_500",
    "train_500",
]
test_split_names = ["test"]
repo_id_out = "fabiencasenave/Tensile2d"


constant_features = sorted(
    [
        "Base_2_2",
        "Base_2_2/Tensile2d",
        "Base_2_2/Zone/CellData",
        "Base_2_2/Zone/CellData/GridLocation",
        "Base_2_2/Zone/Elements_TRI_3",
        "Base_2_2/Zone/FamilyName",
        "Base_2_2/Zone/GridCoordinates",
        "Base_2_2/Zone/PointData",
        "Base_2_2/Zone/PointData/GridLocation",
        "Base_2_2/Zone/SurfaceData",
        "Base_2_2/Zone/SurfaceData/GridLocation",
        "Base_2_2/Zone/ZoneBC",
        "Base_2_2/Zone/ZoneBC/Bottom",
        "Base_2_2/Zone/ZoneBC/Bottom/GridLocation",
        "Base_2_2/Zone/ZoneBC/BottomLeft",
        "Base_2_2/Zone/ZoneBC/BottomLeft/GridLocation",
        "Base_2_2/Zone/ZoneBC/Top",
        "Base_2_2/Zone/ZoneBC/Top/GridLocation",
        "Base_2_2/Zone/ZoneType",
        "Global",
    ]
)

input_features = sorted(
    [
        "Base_2_2/Zone",
        "Base_2_2/Zone/Elements_TRI_3/ElementConnectivity",
        "Base_2_2/Zone/Elements_TRI_3/ElementRange",
        "Base_2_2/Zone/GridCoordinates/CoordinateX",
        "Base_2_2/Zone/GridCoordinates/CoordinateY",
        "Base_2_2/Zone/ZoneBC/Bottom/PointList",
        "Base_2_2/Zone/ZoneBC/BottomLeft/PointList",
        "Base_2_2/Zone/ZoneBC/Top/PointList",
        "Global/P",
        "Global/p1",
        "Global/p2",
        "Global/p3",
        "Global/p4",
        "Global/p5",
    ]
)

output_features = sorted(
    [
        "Base_2_2/Zone/PointData/U1",
        "Base_2_2/Zone/PointData/U2",
        "Base_2_2/Zone/PointData/q",
        "Base_2_2/Zone/PointData/sig11",
        "Base_2_2/Zone/PointData/sig12",
        "Base_2_2/Zone/PointData/sig22",
        "Global/max_U2_top",
        "Global/max_q",
        "Global/max_sig22_top",
        "Global/max_von_mises",
    ]
)

constant_features_benchmark = sorted(
    [
        "Base_2_2",
        "Base_2_2/Tensile2d",
        "Base_2_2/Zone/CellData",
        "Base_2_2/Zone/CellData/GridLocation",
        "Base_2_2/Zone/Elements_TRI_3",
        "Base_2_2/Zone/FamilyName",
        "Base_2_2/Zone/GridCoordinates",
        "Base_2_2/Zone/PointData",
        "Base_2_2/Zone/PointData/GridLocation",
        "Base_2_2/Zone/SurfaceData",
        "Base_2_2/Zone/SurfaceData/GridLocation",
        "Base_2_2/Zone/ZoneBC",
        "Base_2_2/Zone/ZoneBC/Bottom",
        "Base_2_2/Zone/ZoneBC/Bottom/GridLocation",
        "Base_2_2/Zone/ZoneBC/BottomLeft",
        "Base_2_2/Zone/ZoneBC/BottomLeft/GridLocation",
        "Base_2_2/Zone/ZoneBC/Top",
        "Base_2_2/Zone/ZoneBC/Top/GridLocation",
        "Base_2_2/Zone/ZoneType",
        "Global",
    ]
)

input_features_benchmark = sorted(
    [
        "Base_2_2/Zone",
        "Base_2_2/Zone/Elements_TRI_3/ElementConnectivity",
        "Base_2_2/Zone/Elements_TRI_3/ElementRange",
        "Base_2_2/Zone/GridCoordinates/CoordinateX",
        "Base_2_2/Zone/GridCoordinates/CoordinateY",
        "Base_2_2/Zone/ZoneBC/Bottom/PointList",
        "Base_2_2/Zone/ZoneBC/BottomLeft/PointList",
        "Base_2_2/Zone/ZoneBC/Top/PointList",
        "Global/P",
        "Global/p1",
        "Global/p2",
        "Global/p3",
        "Global/p4",
        "Global/p5",
    ]
)

output_features_benchmark = sorted(
    [
        "Base_2_2/Zone/PointData/U1",
        "Base_2_2/Zone/PointData/U2",
        "Base_2_2/Zone/PointData/sig11",
        "Base_2_2/Zone/PointData/sig12",
        "Base_2_2/Zone/PointData/sig22",
        "Global/max_U2_top",
        "Global/max_sig22_top",
        "Global/max_von_mises",
    ]
)


# repo_id = "PLAID-datasets/VKI-LS59"
# split_names = ["train", "test"]
# repo_id_out = "fabiencasenave/VKI-LS59"

# repo_id = "PLAID-datasets/2D_profile"
# split_names = ["train", "test"]
# repo_id_out = "fabiencasenave/2D_profile"

# repo_id = "PLAID-datasets/Rotor37"
# split_names = ["train_1000", "test"]
# repo_id_out = "fabiencasenave/Rotor37"

# repo_id = "PLAID-datasets/2D_Multiscale_Hyperelasticity"
# split_names = ["DOE_train", "DOE_test"]
# repo_id_out = "fabiencasenave/2D_Multiscale_Hyperelasticity"

# repo_id = "PLAID-datasets/2D_ElastoPlastoDynamics"
# split_names = ["train", "test"]
# repo_id_out = "fabiencasenave/2D_ElastoPlastoDynamics"

# ---------------------------------------------------------------------------------------

# load existing binary dataset
hf_dataset = huggingface_bridge.load_dataset_from_hub(repo_id, split="all_samples")
pb_def_ = huggingface_bridge.huggingface_description_to_problem_definition(
    hf_dataset.description
)
infos = huggingface_bridge.huggingface_description_to_infos(hf_dataset.description)


n_samples = len(hf_dataset)


all_pb_def = []
for train_split_name, pb_def_name in zip(train_split_names, pb_def_names):
    pb_def = ProblemDefinition()
    if "benchmark" not in pb_def_name:
        pb_def.add_in_features_identifiers(input_features)
        pb_def.add_out_features_identifiers(output_features)
        pb_def.add_cte_features_identifiers(constant_features)
    else:
        pb_def.add_in_features_identifiers(input_features_benchmark)
        pb_def.add_out_features_identifiers(output_features_benchmark)
        pb_def.add_cte_features_identifiers(constant_features_benchmark)
    # pb_def.set_split(pb_def_.get_split())
    pb_def.set_task(pb_def_.get_task())
    pb_def.set_score_function("RRMSE")

    train_ids = [
        pb_def_.get_split(split_names[0]).index(i)
        for i in pb_def_.get_split(train_split_name)
    ]

    pb_def.set_train_split({split_names_out[0]: train_ids})
    _test_split = {}
    for sn in test_split_names:
        _test_split[sn] = "all"
    pb_def.set_test_split(_test_split)
    all_pb_def.append(pb_def)


# modification for 2D_ElastoPlastoDynamics (suppression of cgns_links)
def update_sample(sample: Sample, split_name: str):
    """Update a PLAID Sample object for the 2D_ElastoPlastoDynamics dataset.

    Modifies the CGNS tree structure within the sample to:
      - Remove specific nodes (e.g., SurfaceData, Bulk).
      - Rename and reorganize fields for compatibility.
      - Copy mesh base data across time steps.
      - Adjust field names and locations based on the split ("train" or others).

    Args:
        sample (Sample): The PLAID Sample to update.
        split_name (str): The split name ("train", "test", etc.) used to determine field modifications.

    Returns:
        Sample: The updated Sample object with modified CGNS tree structure.
    """
    times = sample.get_all_mesh_times()

    mesh_0 = sample.features.data[times[0]]
    flat_mesh_0, cgns_types_0 = flatten_cgns_tree(mesh_0)
    flat_mesh_0.pop("Base_2_2/Zone/SurfaceData")
    flat_mesh_0.pop("Base_2_2/Zone/SurfaceData/GridLocation")

    flat_mesh_0 = {k: flat_mesh_0[k] for k in sorted(flat_mesh_0.keys())}

    sample.features.data[0] = unflatten_cgns_tree(flat_mesh_0, cgns_types_0)

    flat_mesh_base = {
        k: v
        for k, v in flat_mesh_0.items()
        if ("FamilyName" in k) or ("GridCoordinates" in k) or ("Elements_TRI_3" in k)
    }

    for t in times[1:]:
        mesh_1 = sample.features.data[t]
        flat_mesh_1, _ = flatten_cgns_tree(mesh_1)

        if split_name == "train":
            flat_mesh_1["Base_2_2/Zone/PointData"] = flat_mesh_1[
                "Base_2_2/Zone/VertexFields"
            ]
            flat_mesh_1["Base_2_2/Zone/PointData/GridLocation"] = flat_mesh_1[
                "Base_2_2/Zone/VertexFields/GridLocation"
            ]
            flat_mesh_1.pop("Base_2_2/Zone/VertexFields")
            flat_mesh_1.pop("Base_2_2/Zone/VertexFields/GridLocation")

            flat_mesh_1["Base_2_2/Zone/PointData/U_x"] = flat_mesh_1[
                "Base_2_2/Zone/VertexFields/U_x"
            ]
            flat_mesh_1["Base_2_2/Zone/PointData/U_y"] = flat_mesh_1[
                "Base_2_2/Zone/VertexFields/U_y"
            ]
            flat_mesh_1.pop("Base_2_2/Zone/VertexFields/U_x")
            flat_mesh_1.pop("Base_2_2/Zone/VertexFields/U_y")

            flat_mesh_1["Base_2_2/Zone/CellData"] = flat_mesh_1[
                "Base_2_2/Zone/CellCenterFields"
            ]
            flat_mesh_1["Base_2_2/Zone/CellData/GridLocation"] = flat_mesh_1[
                "Base_2_2/Zone/CellCenterFields/GridLocation"
            ]
            flat_mesh_1["Base_2_2/Zone/CellData/EROSION_STATUS"] = flat_mesh_1[
                "Base_2_2/Zone/CellCenterFields/EROSION_STATUS"
            ]

            flat_mesh_1.pop("Base_2_2/Zone/CellCenterFields")
            flat_mesh_1.pop("Base_2_2/Zone/CellCenterFields/GridLocation")
            flat_mesh_1.pop("Base_2_2/Zone/CellCenterFields/EROSION_STATUS")

        flat_mesh_1["Base_2_2/2D_ElastoPlastoDynamics"] = flat_mesh_0[
            "Base_2_2/2D_ElastoPlastoDynamics"
        ]
        flat_mesh_1.pop("Base_2_2/Bulk")

        flat_mesh_1.update(flat_mesh_base)

        flat_mesh_1 = {k: flat_mesh_1[k] for k in sorted(flat_mesh_1.keys())}

        sample.features.data[t] = unflatten_cgns_tree(flat_mesh_1, cgns_types_0)

    return sample


# out-of-code solution with generator
generators = {}
for split_name, split_name_out in zip(split_names, split_names_out):
    ids = pb_def_.get_split(split_name)  # [:2]

    def _generator(ids=ids, split_name=split_name):
        for id in ids:
            sample = huggingface_bridge.binary_to_plaid_sample(hf_dataset[id])
            if repo_id == "PLAID-datasets/2D_ElastoPlastoDynamics":
                sample = update_sample(sample, split_name)
            yield sample

    generators[split_name_out] = _generator


hf_dataset_dict, flat_cst, key_mappings = (
    huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
        generators, verbose=True
    )
)
cgns_types = key_mappings["cgns_types"]


# update infos
dataset = Dataset()
dataset.set_infos(infos)
infos = dataset.get_infos()


# # push to HF hub
# huggingface_bridge.push_dataset_dict_to_hub(repo_id_out, hf_dataset_dict)
# huggingface_bridge.push_infos_to_hub(repo_id_out, infos)
# huggingface_bridge.push_tree_struct_to_hub(repo_id_out, flat_cst, key_mappings)
# for pb_name, pb_def_iter in zip(pb_def_names, all_pb_def):
#     huggingface_bridge.push_problem_definition_to_hub(repo_id_out, pb_name, pb_def_iter)


# push to disk
local_repo = "Tensile2d"
huggingface_bridge.save_dataset_dict_to_disk(local_repo, hf_dataset_dict)
huggingface_bridge.save_infos_to_disk(local_repo, infos)
huggingface_bridge.save_tree_struct_to_disk(local_repo, flat_cst, key_mappings)
for pb_name, pb_def_iter in zip(pb_def_names, all_pb_def):
    huggingface_bridge.save_problem_definition_to_disk(local_repo, pb_name, pb_def_iter)


# sanity check (not working with 2D_ElastoPlastoDynamics)
for split_name, split_name_out in zip(split_names, split_names_out):
    start = time()
    dataset_2 = huggingface_bridge.to_plaid_dataset(
        hf_dataset_dict[split_name_out],
        flat_cst[split_name_out],
        cgns_types,
        enforce_shapes=True,
    )
    print(f"Duration initialization dataset = {time() - start}")

    ii = 1
    ind = pb_def_.get_split(split_name)[ii]

    sample_ind = huggingface_bridge.binary_to_plaid_sample(hf_dataset[ind])
    if repo_id == "PLAID-datasets/2D_ElastoPlastoDynamics":
        sample_ind = update_sample(sample_ind, split_name)
    sample_ind.save("sample", overwrite=True)
    tree_in = sample_ind.features.data[0.0]
    tree_out = dataset_2[ii].features.data[0.0]

    show_cgns_tree(tree_in)
    print("------------")
    show_cgns_tree(tree_out)

    print(
        "tree equal? =",
        compare_cgns_trees_no_types(
            huggingface_bridge.binary_to_plaid_sample(hf_dataset[ind]).features.data[
                0.0
            ],
            dataset_2[ii].features.data[0.0],
        ),
    )
    print("==========================")

    # uncomment for sample save and visualization on paraview
    # dataset_2[ii].save(f"sample_out_{split_name}", overwrite=True)


# Check reading the pushed dataset
hf_dataset_new = huggingface_bridge.load_dataset_from_hub(repo_id_out)
flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id_out)
pb_def = huggingface_bridge.load_problem_definition_from_hub(
    repo_id_out, pb_def_names[0]
)
infos = huggingface_bridge.load_infos_from_hub(repo_id_out)
cgns_types = key_mappings["cgns_types"]


init_ram = get_mem()
start = time()
sample = huggingface_bridge.to_plaid_sample(
    hf_dataset_new[split_names_out[0]], 0, flat_cst[split_names_out[0]], cgns_types
)
elapsed = time() - start
print(
    f"Time to build first sample of split {split_names_out[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)

print("starting conversion")
init_ram = get_mem()
start = time()
dataset = huggingface_bridge.to_plaid_dataset(
    hf_dataset_new[split_names_out[0]], flat_cst[split_names_out[0]], cgns_types
)
elapsed = time() - start
print(
    f"Time to build dataset on split {split_names_out[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)
