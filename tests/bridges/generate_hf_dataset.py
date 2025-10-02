import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

from plaid.bridges import huggingface_bridge


# DATASET_NAME = "Tensile2d"
# SPLIT_NAMES = ["train_500", "test", "OOD"]

# DATASET_NAME = "Rotor37"
# SPLIT_NAMES = ["train_1000", "test"]

# DATASET_NAME = "VKI-LS59"
# SPLIT_NAMES = ["train", "test"]

# DATASET_NAME = "2D_Multiscale_Hyperelasticity"
# SPLIT_NAMES = ["DOE_train", "DOE_test"]

DATASET_NAME = "2D_profile"
SPLIT_NAMES = ["train", "test"]


if __name__ == "__main__":
    print("Loading hf dataset old")
    hf_dataset = huggingface_bridge.load_dataset_from_hub(
        f"PLAID-datasets/{DATASET_NAME}", split="all_samples", num_proc = 12
    )
    pb_def = huggingface_bridge.huggingface_description_to_problem_definition(
        hf_dataset.description
    )
    infos = huggingface_bridge.huggingface_description_to_infos(hf_dataset.description)

    plaid_dataset = huggingface_bridge.huggingface_dataset_to_plaid_binary(
        hf_dataset, processes_number=12, verbose=True
    )

    # print("flattening trees and infering hf features")
    main_splits = {split_name:pb_def.get_split(split_name) for split_name in SPLIT_NAMES}

    dataset_hf_new, flat_cst, key_mappings = huggingface_bridge.plaid_dataset_to_huggingface(
        plaid_dataset, main_splits, processes_number=12)

    # dir_test = f"{DATASET_NAME}"
    # huggingface_bridge.save_dataset_dict_to_disk(dir_test, dataset_hf_new)
    # huggingface_bridge.save_tree_struct_to_disk(dir_test, flat_cst, key_mappings)
    # huggingface_bridge.save_infos_to_disk(dir_test, infos)
    # huggingface_bridge.save_problem_definition_to_disk(dir_test, "task_1", pb_def)

    repo_id = f"fabiencasenave/{DATASET_NAME}"
    huggingface_bridge.push_dataset_dict_to_hub(repo_id, dataset_hf_new)
    huggingface_bridge.push_tree_struct_to_hub(repo_id, flat_cst, key_mappings)
    huggingface_bridge.push_infos_to_hub(repo_id, infos)
    huggingface_bridge.push_problem_definition_to_hub(repo_id, "task_1", pb_def)
