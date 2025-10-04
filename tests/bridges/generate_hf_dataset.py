import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

from plaid.bridges import huggingface_bridge

DATASET_NAME = "Tensile2d"
SPLIT_NAMES = ["train_500", "test", "OOD"]

# DATASET_NAME = "Rotor37"
# SPLIT_NAMES = ["train_1000", "test"]

# DATASET_NAME = "VKI-LS59"
# SPLIT_NAMES = ["train", "test"]

# DATASET_NAME = "2D_Multiscale_Hyperelasticity"
# SPLIT_NAMES = ["DOE_train", "DOE_test"]

# DATASET_NAME = "2D_profile"
# SPLIT_NAMES = ["train", "test"]


if __name__ == "__main__":
    print("Reading hf dataset old format")
    hf_dataset = huggingface_bridge.load_dataset_from_hub(
        f"PLAID-datasets/{DATASET_NAME}", split="all_samples", num_proc=12
    )

    plaid_dataset, pb_def = huggingface_bridge.huggingface_dataset_to_plaid(
        hf_dataset, processes_number=1, verbose=True
    )

    # print("flattening trees and infering hf features")
    main_splits = {
        split_name: pb_def.get_split(split_name) for split_name in SPLIT_NAMES
    }

    # hf_datasetdict_new, flat_cst, key_mappings = (
    #     huggingface_bridge.plaid_dataset_to_huggingface_datasetdict(
    #         plaid_dataset, main_splits, processes_number=12
    #     )
    # )

    generators = {}
    for split_name, ids in main_splits.items():

        def generator_(ids=ids):
            for id in ids:
                yield plaid_dataset[
                    id
                ]  # can be replaced by real out-of-core (read solution from disk, compute the solution)

        generators[split_name] = generator_

    hf_datasetdict_new, flat_cst, key_mappings = (
        huggingface_bridge.plaid_generator_to_huggingface_datasetdict(generators)
    )

    # dir_test = f"{DATASET_NAME}"
    # huggingface_bridge.save_dataset_dict_to_disk(dir_test, hf_datasetdict_new)
    # huggingface_bridge.save_tree_struct_to_disk(dir_test, flat_cst, key_mappings)
    # huggingface_bridge.save_infos_to_disk(dir_test, infos)
    # huggingface_bridge.save_problem_definition_to_disk(dir_test, "task_1", pb_def)

    repo_id = f"fabiencasenave/{DATASET_NAME}"
    huggingface_bridge.push_dataset_dict_to_hub(repo_id, hf_datasetdict_new)
    huggingface_bridge.push_tree_struct_to_hub(repo_id, flat_cst, key_mappings)
    huggingface_bridge.push_infos_to_hub(repo_id, plaid_dataset.get_infos())
    huggingface_bridge.push_problem_definition_to_hub(repo_id, "task_1", pb_def)
