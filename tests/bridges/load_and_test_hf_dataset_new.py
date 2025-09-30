from time import time

import yaml
from huggingface_hub import hf_hub_download

from plaid.bridges import huggingface_bridge
from plaid.utils.base import get_mem
from plaid.utils.cgns_helper import (
    compare_cgns_trees,
    flatten_cgns_tree,
    flatten_cgns_tree_optree,
    unflatten_cgns_tree,
    unflatten_cgns_tree_optree,
)

if __name__ == "__main__":
    print("Initializations:")

    hf_dataset_old = huggingface_bridge.load_hf_dataset_from_hub(
        "PLAID-datasets/Tensile2d", split="all_samples"
    )

    repo_id = "fabiencasenave/Tensile2d_test2"
    infos = huggingface_bridge.load_hf_infos_from_hub(repo_id)
    pb_def = huggingface_bridge.load_hf_problem_definition_from_hub(repo_id, "task_1")

    train_id = range(500)

    yaml_path = hf_hub_download(
        repo_id=repo_id, filename="key_mappings.yaml", repo_type="dataset"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        key_mappings = yaml.safe_load(f)

    fn = key_mappings["features_names"]
    fn["P"] = "Global/P"
    dtypes = key_mappings["dtypes"]
    cgns_types = key_mappings["cgns_types"]

    fnn = list(fn.keys())

    print()
    print("Experience 1: zero copy columnar retrieval")
    print()

    start = time()
    hf_dataset_new = huggingface_bridge.load_hf_dataset_from_hub(
        repo_id, split="train_500"
    )
    end = time()
    print("Time to instanciate cached HF dataset =", end - start)

    print("Initial RAM usage:", get_mem(), "MB")
    start = time()
    all_data = {}
    for i in range(len(hf_dataset_new)):
        for n in fnn:
            all_data[(i, n)] = hf_dataset_new.data[fn[n]][i].values.to_numpy(
                zero_copy_only=True
            )
    end = time()
    print("Time to initiate numpy objects for all the data =", end - start)
    print("RAM usage after loop:", get_mem(), "MB")

    print("check retrieval: sig11=", all_data[(256, "sig11")])

    print()

    # arrow_table = hf_dataset_new.data  # this is a pyarrow.Table
    # arrow_table = arrow_table.select([fn["P"], fn["sig11"]])

    print("Experience 2: plaid dataset generation from HF dataset: old vs new")
    print()

    start = time()
    plaid_dataset = huggingface_bridge.huggingface_dataset_to_plaid(
        hf_dataset_old, ids=train_id, processes_number=1, verbose=True
    )
    end = time()
    print("binary blob conversion plaid dataset generation =", end - start)

    # tree = plaid_dataset[0].features.data[0]

    # leaves, treedef, data_dict, cgns_types_dict = flatten_cgns_tree_optree(tree)
    # # print(leaves[0], leaves[1], leaves[2], leaves[3], leaves[4])
    # # print(type(leaves))
    # print(treedef)
    # print(type(treedef))
    # start = time()
    # for _ in range(1000):
    #     unflat = unflatten_cgns_tree_optree(leaves, treedef, data_dict, cgns_types_dict)
    # end = time()
    # print("1000 unflatten_cgns_tree_optree duration =", end - start)

    # flat, dtypes, cgns_types = flatten_cgns_tree(tree)
    # start = time()
    # for _ in range(1000):
    #     unflat = unflatten_cgns_tree(flat, dtypes, cgns_types)
    # end = time()
    # print("1000 unflatten_cgns_tree duration =", end - start)

    # print(
    #     "first sample CGNS trees identical?:",
    #     compare_cgns_trees(tree, unflat),
    # )

    # # show_cgns_tree(tree)
    # # print("--------------")
    # # show_cgns_tree(unflat)


    # 1.0 / 0.0

    start = time()
    plaid_dataset_new = huggingface_bridge.huggingface_dataset_to_plaid_new(
        hf_dataset_new, dtypes, cgns_types, processes_number=12
    )
    end = time()

    print("tree deflatenning plaid dataset generation =", end - start)

    print(
        "first sample CGNS trees identical?:",
        compare_cgns_trees(
            plaid_dataset[0].features.data[0], plaid_dataset_new[0].features.data[0]
        ),
    )

    1.0 / 0.0

    print()
    print("Experience 3: new HF dataset streaming retrieval time")
    print()

    start = time()
    hf_dataset_test = huggingface_bridge.load_hf_dataset_from_hub(
        repo_id, split="train_500", streaming=True
    )
    hf_dataset_col = hf_dataset_test.select_columns(list(fn.values()))
    print("Streaming hf dataset new")
    for sample in hf_dataset_col:
        for n in fnn:
            sample[fn[n]]
    end = time()
    print("Duration streaming retrieval =", end - start)
