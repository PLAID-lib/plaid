"""Conversion scripts for binary to arrow native types for PLAID-datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from time import time

from plaid import Sample
from plaid.bridges import huggingface_bridge
from plaid.utils.base import get_mem
from plaid.utils.cgns_helper import (
    compare_cgns_trees_no_types,
    flatten_cgns_tree,
    show_cgns_tree,
    unflatten_cgns_tree,
)

# choose repo to convert
repo_id = "PLAID-datasets/Tensile2d"
split_names = ["train_500", "test", "OOD"]
repo_id_out = "fabiencasenave/Tensile2d"

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
infos = huggingface_bridge.huggingface_description_to_infos(hf_dataset.description)
pb_def = huggingface_bridge.huggingface_description_to_problem_definition(
    hf_dataset.description
)


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
for split_name in split_names:
    ids = pb_def.get_split(split_name)  # [:2]

    def _generator(ids=ids, split_name=split_name):
        for id in ids:
            sample = huggingface_bridge.binary_to_plaid_sample(hf_dataset[id])
            if repo_id == "PLAID-datasets/2D_ElastoPlastoDynamics":
                sample = update_sample(sample, split_name)
            yield sample

    generators[split_name] = _generator


hf_dataset_dict, flat_cst, key_mappings = (
    huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
        generators, verbose=True
    )
)
cgns_types = key_mappings["cgns_types"]


# push to HF hub
huggingface_bridge.push_dataset_dict_to_hub(repo_id_out, hf_dataset_dict)
huggingface_bridge.push_infos_to_hub(repo_id_out, infos)
huggingface_bridge.push_tree_struct_to_hub(repo_id_out, flat_cst, key_mappings)
huggingface_bridge.push_problem_definition_to_hub(repo_id_out, "task_1", pb_def)


# sanity check (not working with 2D_ElastoPlastoDynamics)

for split_name in split_names:
    start = time()
    dataset_2 = huggingface_bridge.to_plaid_dataset(
        hf_dataset_dict[split_name],
        flat_cst[split_name],
        cgns_types,
        enforce_shapes=True,
    )
    print(f"Duration initialization dataset = {time() - start}")

    ii = 1
    ind = pb_def.get_split(split_name)[ii]

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
pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id_out, "task_1")
infos = huggingface_bridge.load_infos_from_hub(repo_id_out)
cgns_types = key_mappings["cgns_types"]


init_ram = get_mem()
start = time()
sample = huggingface_bridge.to_plaid_sample(
    hf_dataset_new[split_names[0]], 0, flat_cst[split_names[0]], cgns_types
)
elapsed = time() - start
print(
    f"Time to build first sample of split {split_names[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)

print("starting conversion")
init_ram = get_mem()
start = time()
dataset = huggingface_bridge.to_plaid_dataset(
    hf_dataset_new[split_names[0]], flat_cst[split_names[0]], cgns_types
)
elapsed = time() - start
print(
    f"Time to build dataset on split {split_names[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
)
