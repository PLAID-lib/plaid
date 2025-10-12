from time import time

import numpy as np
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import ConstantRectilinearMeshTools as CRMT

from plaid import Dataset, Sample
from plaid.bridges import huggingface_bridge
from plaid.utils.cgns_helper import show_cgns_tree

geometry = np.load("/mnt/e/input_geometry.npz")

file_X = np.load("/mnt/e/Re_841.npz")
X = file_X["data"]
print("X.shape =", X.shape)

N_x = 1024
N_y = 256
mesh = CRMT.CreateConstantRectilinearMesh(
    dimensions=[N_x, N_y],
    origin=[0.0, 0.0],
    spacing=[64.0 / (N_x - 1), 16.0 / (N_y - 1)],
)

# base_tree = MeshToCGNS(mesh, exportOriginalIDs=False)

samples_list = []
for i in range(1):  # X.shape[0]):
    sample = Sample()
    for t in range(2, 10):  # X.shape[0]):
        sample.features.add_tree(
            MeshToCGNS(mesh, exportOriginalIDs=False), time=float(t)
        )
        for j in range(X.shape[3]):
            field = X[t, :, :, j].T.flatten()
            sample.features.add_field(f"field_{j + 1}", field, time=float(t))
        if t == 2:
            sample.features.add_field(
                "sdf", geometry["data"].T.flatten(), time=float(t)
            )
            sample.features.add_field(
                "mask", geometry["mask"].T.flatten(), time=float(t)
            )

    samples_list.append(sample)

print("samples_list =", samples_list)


dataset = Dataset(samples=samples_list)

show_cgns_tree(dataset[0].features.data[2])
# 1./0.

print(dataset)

dataset[0].save("/mnt/e/Re_841", overwrite=True)
print("saved")

# dataset._save_to_dir_("/mnt/e/LDC_3d")

split_names = ["all_samples"]


def generator_():
    for sample in dataset:
        yield sample


generators = {"all_samples": generator_}


hf_dataset_dict, key_mappings, flat_cst = huggingface_bridge.generate_huggingface_time(
    generators, verbose=True
)
cgns_types = key_mappings["cgns_types"]
print("flat_cst.keys() =", flat_cst.keys())


# start = time()
# dataset_2 = huggingface_bridge.to_plaid_dataset_time(hf_dataset_dict[split_names[0]], cgns_types, flat_cst, enforce_shapes=True)
# print(f"Duration initialization dataset = {time() - start}")

1.0 / 0.0


# file_X = np.load("/mnt/e/LDC_3d_X.npz")
# data_X = file_X["data"]
# np.save("/mnt/e/data_X_extract", data_X[:2,:,:,:,:])
# print("data_X:", data_X.shape)

X = np.load("/mnt/e/data_X_extract.npy")
print("X.shape =", X.shape)

# file_Y = np.load("/mnt/e/LDC_3d_Y.npz")
# data_Y = file_Y["data"]
# np.save("/mnt/e/data_Y_extract", data_Y[:2,:,:,:,:])
# print("data_Y:", data_Y.shape)

Y = np.load("/mnt/e/data_Y_extract.npy")
print("Y.shape =", Y.shape)

N = X.shape[2]
mesh = CRMT.CreateConstantRectilinearMesh(
    dimensions=[N, N, N],
    origin=[0.0, 0.0, 0.0],
    spacing=[1.0 / (N - 1), 1.0 / (N - 1), 1.0 / (N - 1)],
)

# base_tree = MeshToCGNS(mesh, exportOriginalIDs=False)

samples_list = []
for i in range(X.shape[0]):
    sample = Sample()
    sample.add_tree(MeshToCGNS(mesh, exportOriginalIDs=False))
    for j in range(X.shape[1]):
        sample.add_field(f"X_{j + 1}", X[i, j, :, :, :].flatten())
    for j in range(Y.shape[1]):
        sample.add_field(f"Y_{j + 1}", Y[i, j, :, :, :].flatten())

    samples_list.append(sample)


dataset = Dataset(samples=samples_list)

print(dataset)

# dataset._save_to_dir_("/mnt/e/LDC_3d")

split_names = ["all_samples"]


def generator_():
    for sample in dataset:
        yield sample


generators = {"all_samples": generator_}


hf_dataset_dict, key_mappings, flat_cst = huggingface_bridge.generate_huggingface_time(
    generators, verbose=True
)
cgns_types = key_mappings["cgns_types"]
# print(hf_dataset_dict)
# print(key_mappings)
# print(flat_cst)


# huggingface_bridge.save_dataset_dict_to_disk("/mnt/e/LDC_3d", hf_dataset_dict)
# # huggingface_bridge.save_infos_to_disk("/mnt/e/LDC_3d", infos)
# huggingface_bridge.save_tree_struct_to_disk("/mnt/e/LDC_3d", flat_cst, key_mappings)
# # huggingface_bridge.push_problem_definition_to_hub(repo_id_out, "task_1", pb_def)


start = time()
dataset_2 = huggingface_bridge.to_plaid_dataset_time(
    hf_dataset_dict[split_names[0]], cgns_types, flat_cst, enforce_shapes=True
)
print(f"Duration initialization dataset = {time() - start}")

print("dataset_2 =", dataset_2)

# print(mesh)
# show_cgns_tree()

# file_X = np.load("/mnt/e/harmonics_lid_driven_cavity_X.npz", allow_pickle=True)
# data_X = file_X["data"]
# print("X.shape =", data_X.shape)

# file_Y = np.load("/mnt/e/harmonics_lid_driven_cavity_Y.npz", allow_pickle=True)
# data_Y = file_Y["data"]
# print("X.shape =", data_Y.shape)


# repo_id = "fabiencasenave/Tensile2d"
# split_names = ["train_500", "test", "OOD"]
# field = "U1"
# repo_id_out = "fabiencasenave/Tensile2d_new"

# # repo_id = "fabiencasenave/VKI-LS59"
# # split_names = ["train", "test"]
# # field = "mach"


# hf_dataset_dict = huggingface_bridge.load_dataset_from_hub(repo_id)


# flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
# pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, "task_1")
# infos = huggingface_bridge.load_infos_from_hub(repo_id)
# cgns_types = key_mappings["cgns_types"]

# hf_dataset = hf_dataset_dict[split_names[0]]

# init_ram = get_mem()
# start = time()
# dataset = huggingface_bridge.to_plaid_dataset(hf_dataset, flat_cst, cgns_types)
# elapsed = time() - start
# print(f"Time to build dataset on split {split_names[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem()-init_ram} MB")

# # for id in range(len(dataset)):
# #     sample = dataset[id]
# #     sample.features.data[1.1] = copy.deepcopy(sample.features.data[0.])


# # sample = dataset[0]
# # sample.add_field("sig11", 2.*copy.deepcopy(sample.get_field("sig11", time=1.1)), time=1.1)
# # sample.features.data[2.1] = copy.deepcopy(sample.features.data[0.])

# # print(sample)

# generators = {}
# for split_name in split_names[:1]:
#     ids = pb_def.get_split(split_name)
#     def generator_(ids=ids):
#         for id in ids:
#             yield dataset[id]
#     generators[split_name] = generator_


# hf_dataset_dict, key_mappings, flat_cst = huggingface_bridge.generate_huggingface_time(generators, verbose=True)
# cgns_types = key_mappings["cgns_types"]

# # val = hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/PointData/sig11_value"]
# # first = np.array(val[0:6143])
# # second = np.array(val[6143:2*6143])

# # print("len =", len(val))
# # print("diff =", np.linalg.norm(first-second))
# # print("times =", hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/PointData/sig11_times"])

# huggingface_bridge.push_dataset_dict_to_hub(repo_id_out, hf_dataset_dict)
# huggingface_bridge.push_infos_to_hub(repo_id_out, infos)
# huggingface_bridge.push_tree_struct_to_hub(repo_id_out, flat_cst, key_mappings)
# huggingface_bridge.push_problem_definition_to_hub(repo_id_out, "task_1", pb_def)
# # 1./0.


# start = time()
# dataset_2 = huggingface_bridge.to_plaid_dataset_time(hf_dataset_dict[split_names[0]], cgns_types, flat_cst, enforce_shapes=True)
# print(f"Duration initialization dataset = {time() - start}")

# # times = list(dataset_2[0].features.data.keys())

# # print("1", dataset_2[0].get_field("sig11", time=0.))
# # print("2", dataset_2[0].get_field("sig11", time=1.1))

# tree_in = dataset[0].features.data[0.]
# tree_out = dataset_2[0].features.data[0.]

# show_cgns_tree(tree_in)
# print("------------")
# show_cgns_tree(tree_out)

# print("tree equal? =", compare_cgns_trees_no_types(dataset[0].features.data[0.], dataset_2[0].features.data[0.]))
# # print("tree equal? =", compare_cgns_trees_no_types(dataset[0].features.data[1.1], dataset_2[0].features.data[1.1]))
# # print("tree equal? =", compare_cgns_trees_no_types(dataset[0].features.data[2.1], dataset_2[0].features.data[2.1]))

# dataset_2[0].save("sample_out", overwrite=True)

# # print(dataset_2[0].features.data[times[0]])
# # print(dataset_2[0].features.data[times[1]])

# 1./0.
