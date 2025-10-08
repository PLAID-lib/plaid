
from plaid.bridges import huggingface_bridge
from plaid import Dataset, Sample, ProblemDefinition
from plaid.types import FeatureIdentifier
from plaid.utils.base import get_mem

from Muscat.Bridges.CGNSBridge import CGNSToMesh, MeshToCGNS

from time import time
import numpy as np
import copy

from plaid.utils.cgns_helper import compare_cgns_trees, compare_cgns_trees_no_types, show_cgns_tree


repo_id = "fabiencasenave/Tensile2d"
split_names = ["train_500", "test", "OOD"]
field = "U1"
repo_id_out = "fabiencasenave/Tensile2d_time"

# repo_id = "fabiencasenave/VKI-LS59"
# split_names = ["train", "test"]
# field = "mach"


hf_dataset_dict = huggingface_bridge.load_dataset_from_hub(repo_id)



flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, "task_1")
infos = huggingface_bridge.load_infos_from_hub(repo_id)
cgns_types = key_mappings["cgns_types"]

hf_dataset = hf_dataset_dict[split_names[0]]

init_ram = get_mem()
start = time()
dataset = huggingface_bridge.to_plaid_dataset(hf_dataset, flat_cst, cgns_types)
elapsed = time() - start
print(f"Time to build dataset on split {split_names[0]}: {elapsed:.6g} s, RAM usage increase: {get_mem()-init_ram} MB")

for id in range(len(dataset)):
    sample_0 = dataset[id]
    sample_0.features.data[1.1] = copy.deepcopy(sample_0.features.data[0.])

print(sample_0)

generators = {}
for split_name in split_names[:1]:
    ids = pb_def.get_split(split_name)
    def generator_(ids=ids):
        for id in ids:
            yield dataset[id]
    generators[split_name] = generator_



hf_dataset_dict, cgns_types, flat_cst = huggingface_bridge.generate_huggingface_time(generators, verbose=True)

# huggingface_bridge.push_dataset_dict_to_hub(repo_id_out, hf_dataset_dict)
# print(">>>", hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/ZoneBC/Top/GridLocation_value"])
# print(">>>", hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/ZoneBC/Top/GridLocation_times"])

start = time()
dataset_2 = huggingface_bridge.to_plaid_dataset_time(hf_dataset_dict[split_names[0]], cgns_types, flat_cst, enforce_shapes=True)
print(f"Duration initialization dataset = {time() - start}")

# times = list(dataset_2[0].features.data.keys())

# print("1", dataset_2[0].get_field("sig11", time=0.))
# print("2", dataset_2[0].get_field("sig11", time=1.1))

tree_in = dataset[0].features.data[1.1]
tree_out = dataset_2[0].features.data[1.1]

show_cgns_tree(tree_in)
print("------------")
show_cgns_tree(tree_out)

print("tree equal? =", compare_cgns_trees_no_types(tree_in, tree_out))

dataset_2[0].save("sample_out", overwrite=True)

# print(dataset_2[0].features.data[times[0]])
# print(dataset_2[0].features.data[times[1]])

1./0.

print(hf_dataset_dict)

print(">>", hf_dataset_dict[split_names[0]][0]["Base_2_2/Zone/PointData_times"])

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



