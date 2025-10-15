from time import time

from plaid.bridges import huggingface_bridge
from plaid.utils.base import get_mem
from plaid.utils.cgns_helper import (
    compare_cgns_trees_no_types,
    flatten_cgns_tree,
    show_cgns_tree,
    unflatten_cgns_tree,
)

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

hf_dataset = huggingface_bridge.load_dataset_from_hub(repo_id, split="all_samples")
infos = huggingface_bridge.huggingface_description_to_infos(hf_dataset.description)
pb_def = huggingface_bridge.huggingface_description_to_problem_definition(
    hf_dataset.description
)

# init_ram = get_mem()
# start = time()
# dataset, pb_def = huggingface_bridge.huggingface_dataset_to_plaid(
#     hf_dataset, processes_number=12
# )
# elapsed = time() - start
# print(
#     f"Time to build dataset: {elapsed:.6g} s, RAM usage increase: {get_mem() - init_ram} MB"
# )

# generators = {}
# for split_name in split_names:
#     ids = pb_def.get_split(split_name)

#     def generator_(ids=ids):
#         for id in ids:
#             yield dataset[id]

#     generators[split_name] = generator_


def update_sample(sample, split_name):
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


# sample = huggingface_bridge.binary_to_plaid_sample(hf_dataset[0])
# times = sample.get_all_mesh_times()
# sample = update_sample(sample, "train")
# show_cgns_tree(sample.features.data[times[0]])
# print("-------------")
# show_cgns_tree(sample.features.data[times[-1]])
# sample.save("sample", overwrite=True)
# 1./0.


generators = {}
for split_name in split_names:
    ids = pb_def.get_split(split_name)  # [:2]

    def generator_(ids=ids, split_name=split_name):
        for id in ids:
            sample = huggingface_bridge.binary_to_plaid_sample(hf_dataset[id])
            if repo_id == "PLAID-datasets/2D_ElastoPlastoDynamics":
                sample = update_sample(sample, split_name)
                # show_cgns_tree(sample.features.data[0.01])
                # print(split_name)
                # 1./0.
            yield sample

    generators[split_name] = generator_


hf_dataset_dict, flat_cst, key_mappings = (
    huggingface_bridge.plaid_generator_to_huggingface_datasetdict(
        generators, verbose=True
    )
)
cgns_types = key_mappings["cgns_types"]


# val = hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/PointData/sig11_value"]
# first = np.array(val[0:6143])
# second = np.array(val[6143:2*6143])

# print("len =", len(val))
# print("diff =", np.linalg.norm(first-second))
# print("times =", hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/PointData/sig11_times"])

# huggingface_bridge.save_tree_struct_to_disk("./", flat_cst, key_mappings)

huggingface_bridge.push_dataset_dict_to_hub(repo_id_out, hf_dataset_dict)
huggingface_bridge.push_infos_to_hub(repo_id_out, infos)
huggingface_bridge.push_tree_struct_to_hub(repo_id_out, flat_cst, key_mappings)
huggingface_bridge.push_problem_definition_to_hub(repo_id_out, "task_1", pb_def)


for split_name in split_names:
    start = time()
    dataset_2 = huggingface_bridge.to_plaid_dataset(
        hf_dataset_dict[split_name],
        flat_cst[split_name],
        cgns_types,
        enforce_shapes=True,
    )
    print(f"Duration initialization dataset = {time() - start}")

    # times = list(dataset_2[0].features.data.keys())

    # print("1", dataset_2[0].get_field("sig11", time=0.))
    # print("2", dataset_2[0].get_field("sig11", time=1.1))

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

    dataset_2[ii].save(f"sample_out_{split_name}", overwrite=True)
# print("tree equal? =", compare_cgns_trees_no_types(dataset[0].features.data[1.1], dataset_2[0].features.data[1.1]))
# print("tree equal? =", compare_cgns_trees_no_types(dataset[0].features.data[2.1], dataset_2[0].features.data[2.1]))


# print(dataset_2[0].features.data[times[0]])
# print(dataset_2[0].features.data[times[1]])


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


1.0 / 0.0

# print(hf_dataset_dict)

# print(">>", hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/PointData_times"])

# 1./0.

# # sample_0.add_tree(copy.deepcopy(sample_0.get_mesh()), time=1.)

# mesh = CGNSToMesh(sample_0.get_mesh())
# flat_tree, cgns_types = huggingface_bridge.flatten_cgns_tree(sample_0.get_mesh())

# # keys_to_remove = []
# # for key in flat_tree.keys():
# #     if "ZoneBC" in key or "Elements" in key or "GridCoordinates" in key:
# #         keys_to_remove.append(key)

# flat_tree2 = {k: v for k, v in flat_tree.items()}

# keyfield = [k for k in flat_tree.keys() if field in k][0]
# keyTime = [k for k in flat_tree.keys() if "TimeValues" in k]
# flat_tree2[keyfield] = 2.*flat_tree2[keyfield]
# for t in keyTime:
#     print(flat_tree2[t])
#     flat_tree2[t] = np.array([1.])
# print("keyTime =", keyTime)

# tree = huggingface_bridge.unflatten_cgns_tree(flat_tree2, cgns_types)
# sample_0.add_tree(tree, time=1.)

# # 1./0.

# # mesh.nodeFields["U1"]*=2.

# # # sample_0.init_base(2, 2, base_name = "Base_2_2", time=1.)
# # # # sample_0.init_zone(time=1.)
# # # sample_0.add_field("U1", 2.*sample_0.get_field("U1"), time=1.)
# # sample_0.add_tree(MeshToCGNS(mesh), time=1.)

# sample_0.save("sample", overwrite=True)

# sample_0.show_tree(0.)
# print("-----------------")
# sample_0.show_tree(1.)
