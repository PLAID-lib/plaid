import io
import os

import yaml
from datasets import Dataset, DatasetDict, Features
from huggingface_hub import HfApi

os.environ["HF_HUB_DISABLE_XET"] = "1"

from tqdm import tqdm

from plaid.bridges import huggingface_bridge
from plaid.utils.base import update_dict_only_new_keys
from plaid.utils.cgns_helper import (
    flatten_cgns_tree,
)

if __name__ == "__main__":
    print("Loading hf dataset old")
    hf_dataset = huggingface_bridge.load_hf_dataset_from_hub(
        "PLAID-datasets/Tensile2d", split="all_samples"
    )
    print("loaded")
    pb_def = huggingface_bridge.huggingface_description_to_problem_definition(
        hf_dataset.description
    )
    infos = huggingface_bridge.huggingface_description_to_infos(hf_dataset.description)

    all_feat_names = (
        pb_def.get_input_scalars_names()
        + pb_def.get_output_scalars_names()
        + pb_def.get_output_fields_names()
    )

    print("Converting hf dataset old to plaid dataet")
    plaid_dataset = huggingface_bridge.huggingface_dataset_to_plaid(
        hf_dataset, processes_number=5, verbose=True
    )

    def flat_tree_generator(flat_trees):
        """
        Generator yielding samples from a list of flat_trees.
        Each flat_tree is a dict {key -> value}.
        """
        for ft in flat_trees:
            yield ft

    def make_hf_dataset(flat_tree_list, hf_features):
        """
        Create a HuggingFace dataset from a list of flat_trees and dtypes.
        The features schema is inferred automatically.
        """
        dataset = Dataset.from_generator(
            lambda: flat_tree_generator(flat_tree_list),
            features=Features(hf_features),
        )
        return dataset

    print("flattening trees and infering hf features")

    dtypes = {}
    cgns_types = {}
    hf_features = {}

    flat_tree_list = {}

    split_names = ["train_500", "test", "OOD"]

    for split_name in split_names:
        flat_tree_list[split_name] = []

        for id in tqdm(pb_def.get_split(split_name), desc=f"Processing {split_name}"):
            sample = plaid_dataset[id]
            flat_tree, dtypes_, cgns_types_ = flatten_cgns_tree(sample.features.data[0])
            update_dict_only_new_keys(dtypes, dtypes_)
            update_dict_only_new_keys(cgns_types, cgns_types_)

            hf_features_ = huggingface_bridge.infer_hf_features(flat_tree, dtypes)
            update_dict_only_new_keys(hf_features, hf_features_)

            flat_tree_list[split_name].append(flat_tree)

    # for split_name in split_names:
    #     flat_tree_list[split_name] = []

    #     for id in tqdm(pb_def.get_split(split_name), desc=f"Processing {split_name}"):
    #         sample = plaid_dataset[id]
    #         leaves, treedef = flatten_cgns_tree_optree(sample.features.data[0])
    #         update_dict_only_new_keys(dtypes, dtypes_)
    #         update_dict_only_new_keys(cgns_types, cgns_types_)

    #         hf_features_ = huggingface_bridge.infer_hf_features(flat_tree, dtypes)
    #         update_dict_only_new_keys(hf_features, hf_features_)

    #         flat_tree_list[split_name].append(flat_tree)

    features_names = {}
    for fn in all_feat_names:
        for large_name in cgns_types.keys():
            if "/" + fn in large_name:
                features_names[fn] = large_name
                continue

    print("Pushing key_mappings, pb_def and infos to the hub")

    repo_id = "fabiencasenave/Tensile2d_test2"

    key_mappings = {}
    key_mappings["features_names"] = features_names
    key_mappings["dtypes"] = dtypes
    key_mappings["cgns_types"] = cgns_types

    api = HfApi()
    yaml_str = yaml.dump(key_mappings)
    yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))
    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo="key_mappings.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload key_mappings.yaml",
    )

    huggingface_bridge.push_dataset_infos_to_hub(repo_id, infos)
    huggingface_bridge.push_problem_definition_to_hub(repo_id, "task_1", pb_def)

    print("making hf datasets and pushing to the hub")

    dict_of_hf_datasets = {}
    for split_name in split_names:
        dict_of_hf_datasets[split_name] = make_hf_dataset(
            flat_tree_list[split_name], hf_features
        )

    dset_dict = DatasetDict(dict_of_hf_datasets)
    # huggingface_bridge.push_dataset_dict_to_hub(repo_id, dset_dict)

    # -------------------------------------------------------------------------------------------------------------------
    # SOME TESTS BELOW

    # ds = dset_dict["train_500"]

    # arrow_table = ds.data  # this is a pyarrow.Table
    # arrow_table = arrow_table.select(["Base_2_2/Zone/ZoneBC/Bottom/PointList", "Base_2_2/Zone/PointData/sig11"])

    # list_array = arrow_table["Base_2_2/Zone/PointData/sig11"][0]  # pyarrow.ListArray
    # print("arrow?", type(list_array))
    # print("list?", type(ds[0]["Base_2_2/Zone/PointData/sig11"]))
    # values = list_array.values          # contiguous buffer

    # print()

    # try:
    #     np_values = values.to_numpy(zero_copy_only=True)  # true zero-copy NumPy
    #     print("zero copy retrieval OK!")
    # except:
    #     print("zero copy retrieval not OK!")

    # print()

    # flat_tree0 = huggingface_bridge.reconstruct_flat_tree_from_hf_sample(ds[0], dtypes)
    # unflatten_tree0 = unflatten_cgns_tree(flat_tree0, dtypes, cgns_types)

    # show_cgns_tree(unflatten_tree0)

    # print("trees identical?:", compare_cgns_trees(plaid_dataset[0].features.data[0], unflatten_tree0))

    # from plaid import Sample
    # from plaid.containers.features import SampleFeatures
    # sample = Sample(features=SampleFeatures({0.:unflatten_tree0}))

    # print(sample.get_field("U1"))
    # print(sample.get_scalar("p1"))
