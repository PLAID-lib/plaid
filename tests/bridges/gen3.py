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
    flatten_cgns_tree, flatten_cgns_tree_optree
)

from datasets import Dataset, DatasetDict, Features, Value, Sequence
import numpy as np



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
        hf_dataset, processes_number=1, verbose=True
    )


    # --------------------------
    # Infer HF feature type from actual value
    # --------------------------
    def infer_hf_features_from_value(value):
        if value is None:
            return Value("null")

        # Scalars
        if np.isscalar(value):
            dtype = np.array(value).dtype
            if np.issubdtype(dtype, np.floating):
                return Value("float32")
            elif np.issubdtype(dtype, np.integer):
                return Value("int64")
            elif np.issubdtype(dtype, np.bool_):
                return Value("bool")
            else:
                return Value("string")

        # Arrays / lists
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value)
            base_type = infer_hf_features_from_value(arr.flat[0] if arr.size > 0 else None)
            if arr.ndim == 1:
                return Sequence(base_type)
            elif arr.ndim == 2:
                return Sequence(Sequence(base_type))
            elif arr.ndim == 3:
                return Sequence(Sequence(Sequence(base_type)))
            else:
                raise TypeError(f"Unsupported ndim: {arr.ndim}")
        else:
            return Value("string")

    def collect_schema_from_trees_data(all_trees):
        """
        Collect union of all paths across all trees and produce:
        - global_cgns_types: path → CGNS type
        - hf_features: HuggingFace Features inferred from actual data
        """
        global_cgns_types = {}
        global_types = {}

        for tree in all_trees:
            _, data_dict, cgns_types = flatten_cgns_tree(tree)
            for path, value in data_dict.items():
                # Update CGNS types
                if path not in global_cgns_types:
                    global_cgns_types[path] = cgns_types[path]
                else:
                    # Optional: sanity check for conflicts
                    if global_cgns_types[path] != cgns_types[path]:
                        raise ValueError(
                            f"Conflict for path '{path}': {global_cgns_types[path]} vs {cgns_types[path]}"
                        )

                # Infer HF feature from value
                if path not in global_types:
                    global_types[path] = infer_hf_features_from_value(value)
                # else: already inferred from previous tree

        global_types["treedef"] = Value("string")
        hf_features = Features(global_types)
        return global_cgns_types, hf_features

    def sample_generator(trees, global_cgns_types):
        for tree in trees:
            flat, dtypes, cgns_types = flatten_cgns_tree(tree)
            sample = {path: None for path in global_cgns_types.keys()}
            for path, val in flat.items():
                sample[path] = val
            yield sample


    print("flattening trees and infering hf features")
    split_names = ["train_500", "test", "OOD"]

    all_trees = []
    for split_name in split_names:
        trees_list = [plaid_dataset[id].features.data[0.] for id in pb_def.get_split(split_name)]
        all_trees.extend(trees_list)

    global_cgns_types, hf_features = collect_schema_from_trees_data(all_trees)

    var_path = [k for k,v in global_cgns_types.items() if (v in ["DataArray_t", "IndexArray_t", "Zone_t", "Element_t"] and "Time" not in k)]
    cst_path = [k for k in global_cgns_types.keys() if k not in var_path]


    print("var_path =", var_path)
    print()
    print("cst_path =", cst_path)


    # Build each split
    dict_of_hf_datasets = {}
    for split_name in split_names:
        trees_list = [plaid_dataset[id].features.data[0.] for id in pb_def.get_split(split_name)]
        dict_of_hf_datasets[split_name] = Dataset.from_generator(
            lambda trees=trees_list: sample_generator(trees, global_cgns_types),
            features=hf_features
        )

    dataset_hf_new = DatasetDict(dict_of_hf_datasets)
    dataset_hf_new.save_to_disk("Tensile2d_hf_dataset_new")

    1./0.



    # features_names = {}
    # for fn in all_feat_names:
    #     for large_name in cgns_types.keys():
    #         if "/" + fn in large_name:
    #             features_names[fn] = large_name
    #             continue

    # 1./0.

    # print("Pushing key_mappings, pb_def and infos to the hub")

    # repo_id = "fabiencasenave/Tensile2d_test2"

    # key_mappings = {}
    # key_mappings["features_names"] = features_names
    # key_mappings["dtypes"] = dtypes
    # key_mappings["cgns_types"] = cgns_types

    # api = HfApi()
    # yaml_str = yaml.dump(key_mappings)
    # yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))
    # api.upload_file(
    #     path_or_fileobj=yaml_buffer,
    #     path_in_repo="key_mappings.yaml",
    #     repo_id=repo_id,
    #     repo_type="dataset",
    #     commit_message="Upload key_mappings.yaml",
    # )

    # huggingface_bridge.push_dataset_infos_to_hub(repo_id, infos)
    # huggingface_bridge.push_problem_definition_to_hub(repo_id, "task_1", pb_def)

    print("making hf datasets and pushing to the hub")

    dict_of_hf_datasets = {}
    for split_name in split_names:
        dict_of_hf_datasets[split_name] = make_hf_dataset(
            flat_tree_list[split_name], hf_features
        )

    dset_dict = DatasetDict(dict_of_hf_datasets)

    1./0.

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
