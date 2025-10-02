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
    fix_cgns_tree_types
)


# DATASET_NAME = "Tensile2d"
# SPLIT_NAMES = ["train_500", "test", "OOD"]

DATASET_NAME = "Rotor37"
SPLIT_NAMES = ["train_1000", "test"]


if __name__ == "__main__":
    print("Initializations:")

    start = time()
    hf_dataset_old = huggingface_bridge.load_dataset_from_hub(
        f"PLAID-datasets/{DATASET_NAME}", split="all_samples", num_proc=12
    )
    end = time()
    print("Time to instantiate old HF dataset from hub =", end - start)

    print()
    print("Experience 1: fast columnar retrieval (only 1DArray are instantiated in np-copy mode)")
    print()

    start = time()
    dir_test = f"{DATASET_NAME}_test"
    hf_dataset_new = huggingface_bridge.load_dataset_from_disk(Path(dir_test) / Path(SPLIT_NAMES[0]))
    flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_disk(dir_test)
    pb_def = huggingface_bridge.load_problem_definition_from_disk(dir_test, "task_1")
    infos = huggingface_bridge.load_infos_from_disk(dir_test)

    features_names = key_mappings["variable_features"]
    dtypes = key_mappings["dtypes"]
    cgns_types = key_mappings["cgns_types"]
    end = time()
    print("Time to instantiate new HF dataset from disk =", end - start)


    start = time()
    repo_id = f"fabiencasenave/{DATASET_NAME}_test"
    hf_dataset_new = huggingface_bridge.load_dataset_from_hub(repo_id, split = SPLIT_NAMES[0])
    flat_cst, key_mappings = huggingface_bridge.load_tree_struct_from_hub(repo_id)
    pb_def = huggingface_bridge.load_problem_definition_from_hub(repo_id, "task_1")
    infos = huggingface_bridge.load_infos_from_hub(repo_id)

    features_names = key_mappings["variable_features"]
    dtypes = key_mappings["dtypes"]
    cgns_types = key_mappings["cgns_types"]
    end = time()
    print("Time to instantiate new HF dataset from hub =", end - start)


    print("Initial RAM usage:", get_mem(), "MB")
    start = time()
    all_data = {}
    for i in range(len(hf_dataset_new)):
        for n in features_names:
            all_data[(i, n)] = hf_dataset_new.data[n][i].values.to_numpy(zero_copy_only=False)
    end = time()
    print("Time to initiate numpy objects for all the data =", end - start)
    print("RAM usage after loop:", get_mem(), "MB")

    print(f"check retrieval: {features_names[0]} =", all_data[(0, list(features_names)[0])])

    print()


    print("Experience 2: plaid dataset generation from HF dataset: old vs new")
    print()

    start = time()
    plaid_dataset = huggingface_bridge.huggingface_dataset_to_plaid_binary(
        hf_dataset_old, ids=pb_def.get_split(SPLIT_NAMES[0]), processes_number=12, verbose=True
    )
    end = time()
    print("Time to convert old HF dataset to plaid (binary blobs) =", end - start)

    tic = time()
    plaid_dataset_new = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset_new, flat_cst, cgns_types, dtypes, enforce_type_shapes = False, verbose = False)
    print("Time to convert new HF dataset to plaid (fast) =", time() - tic)

    fix_cgns_tree_types(plaid_dataset[2].features.data[0])
    plaid_dataset[2].save("sample", overwrite=True)

    print(f"get({pb_def.get_output_scalars_names()[0]}): ", plaid_dataset_new[0].get_scalar(pb_def.get_output_scalars_names()[0]))
    print(f"get({pb_def.get_output_fields_names()[0]}): ", plaid_dataset_new[0].get_field(pb_def.get_output_fields_names()[0]))
    print("get(nodes): ", plaid_dataset_new[0].get_nodes())
    print("get(elements): ", plaid_dataset_new[0].get_elements())

    tic = time()
    plaid_dataset_new = huggingface_bridge.huggingface_dataset_to_plaid(hf_dataset_new, flat_cst, cgns_types, dtypes, enforce_type_shapes = True, verbose = False)
    print("Time to convert new HF dataset to plaid (safe) =", time() - tic)


    show_cgns_tree(plaid_dataset[2].features.data[0])
    print("--------------")
    show_cgns_tree(plaid_dataset_new[2].features.data[0])
    print("--------------")


    print(
        "first sample CGNS trees identical (no types)?:",
        compare_cgns_trees_no_types(
            plaid_dataset[2].features.data[0], plaid_dataset_new[2].features.data[0]
        ),
    )


    print()
    print("Experience 3: new HF dataset streaming retrieval time")
    print()

    hf_dataset_test = datasets.load_dataset(
        repo_id, split=SPLIT_NAMES[0], streaming=True
    )
    for sample in tqdm(hf_dataset_test, desc="Streaming hf dataset new"):
        for n in features_names:
            sample[n]
    end = time()
    print("Duration streaming retrieval =", end - start)
