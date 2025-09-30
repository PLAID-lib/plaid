from time import time

import yaml
from huggingface_hub import hf_hub_download

from plaid.bridges import huggingface_bridge
from plaid.utils.base import get_mem
from plaid.utils.cgns_helper import (
    compare_cgns_trees,
    flatten_cgns_tree,
    flatten_cgns_tree_optree,
    show_cgns_tree,
    unflatten_cgns_tree,
    unflatten_cgns_tree_optree,
    flatten_cgns_tree_optree_dict,
    unflatten_cgns_tree_optree_dict
)

from tqdm import tqdm

from plaid import Dataset, ProblemDefinition, Sample
from plaid.containers.features import SampleFeatures

import pickle
import base64



if __name__ == "__main__":

    def deserialize_treedef(serialized_str):
        # Decode base64, then unpickle
        data_bytes = base64.b64decode(serialized_str.encode("utf-8"))
        return pickle.loads(data_bytes)

    print("Initializations:")

    hf_dataset_old = huggingface_bridge.load_hf_dataset_from_hub(
        "PLAID-datasets/Tensile2d", split="all_samples"
    )

    repo_id = "fabiencasenave/Tensile2d_test3"


    train_id = range(500)


    print()
    print("Experience 1: zero copy columnar retrieval")
    print()

    yaml_path = hf_hub_download(
        repo_id="fabiencasenave/Tensile2d_test2", filename="key_mappings.yaml", repo_type="dataset"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        key_mappings = yaml.safe_load(f)

    fn = key_mappings["features_names"]
    fn["P"] = "Global/P"

    fn = {k:"CGNSTree/"+v for k,v in fn.items()}
    fnn = list(fn.keys())

    start = time()
    hf_dataset_new = huggingface_bridge.load_hf_dataset_from_hub(
        repo_id, split="train_500"
    )
    end = time()
    print("Time to instanciate cached HF dataset =", end - start)

    cols = hf_dataset_new.column_names

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

    print()

    print("Experience 2: plaid dataset generation from HF dataset: old vs new")
    print()

    start = time()
    plaid_dataset = huggingface_bridge.huggingface_dataset_to_plaid(
        hf_dataset_old, ids=train_id, processes_number=1, verbose=True
    )
    end = time()
    print("binary blob conversion plaid dataset generation =", end - start)

    tree = plaid_dataset[0].features.data[0]


    # print(treedef)
    # print(type(treedef))
    # start = time()
    # for _ in range(1000):
    #     unflat = unflatten_cgns_tree_optree(leaves, treedef, cgns_types_dict)
    # end = time()
    # print("1000 unflatten_cgns_tree_optree duration =", end - start)


    # leaves, treedef, cgns_types_dict = flatten_cgns_tree_optree(tree)
    # unflat = unflatten_cgns_tree_optree(leaves, treedef, cgns_types_dict)


    treedef, data_dict, cgns_types = flatten_cgns_tree_optree_dict(tree)
    unflat = unflatten_cgns_tree_optree_dict(treedef, data_dict, cgns_types)

    start = time()
    for _ in range(1000):
        unflat = unflatten_cgns_tree_optree_dict(treedef, data_dict, cgns_types)
    end = time()
    print("1000 unflatten_cgns_tree_optree duration =", end - start)

    # show_cgns_tree(tree)
    # print("--------------")
    # show_cgns_tree(unflat)

    print("first sample CGNS trees identical?:", compare_cgns_trees(tree, unflat),)

    flat, dtypes, cgns_types_ = flatten_cgns_tree(tree)
    start = time()
    for _ in range(1000):
        unflat = unflatten_cgns_tree(flat, dtypes, cgns_types_)
    end = time()
    print("1000 unflatten_cgns_tree duration =", end - start)



    hf_dataset_new.set_format("numpy")


    # from concurrent.futures import ProcessPoolExecutor
    # from time import time

    # def treat(idx):
    #     sample = hf_dataset_new[idx]  # pull row inside worker
    #     treedef = deserialize_treedef(str(sample.pop("treedef")))
    #     unflat = unflatten_cgns_tree_optree_dict(treedef, sample, cgns_types)
    #     return Sample(features=SampleFeatures({0.0: unflat}))

    # description = "Building Samples"

    # start = time()
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     # submit tasks for all indices
    #     futures = [executor.submit(treat, i) for i in range(len(hf_dataset_new))]

    #     # iterate results with progress bar
    #     list_of_samples = []
    #     for f in tqdm(futures, desc=description):
    #         list_of_samples.append(f.result())
    # plaid_dataset_new = Dataset(samples=list_of_samples)
    # end = time()

    # print("Parallel time:", end - start)


    description = "Converting Hugging Face dataset to plaid"

    sample_list = []
    t1=t2=t3=0

    start = time()
    # convert once for all samples
    # all_columns = {col: hf_dataset_new[col].to_numpy(zero_copy_only=False) for col in hf_dataset_new.column_names}
    tic = time()
    # hf_dataset_new.set_format("numpy")
    t0 = time()-tic
    for idx in tqdm(range(len(hf_dataset_new)), desc=description):
        tic = time()
        # data_dict = huggingface_bridge.reconstruct_flat_tree_from_hf_sample2(hf_dataset_new, idx)
        # data_dict = hf_dataset_new[idx]#huggingface_bridge.reconstruct_flat_tree_from_hf_sample2(hf_dataset_new[0])
        # data_dict = {key:hf_dataset_new.data[key][idx].values.to_numpy() for key in hf_dataset_new.data.keys()}
        # data_dict = {key: hf_dataset_new.data[key][idx] for key in hf_dataset_new.column_names}
        # data_dict = {col: hf_dataset_new[idx][col] for col in hf_dataset_new.column_names}
        data_dict = hf_dataset_new[idx]
        t1 += time()-tic
        tic = time()
        treedef = deserialize_treedef(str(data_dict.pop("treedef")))
        t2 += time()-tic
        tic = time()
        unflat = unflatten_cgns_tree_optree_dict(treedef, data_dict, cgns_types)
        t3 += time()-tic
        sample_list.append(Sample(features=SampleFeatures({0.0: unflat})))
    plaid_dataset_new = Dataset(samples=sample_list)
    end = time()
    print("indiv times =", t0, t1, t2, t3)
    print("tree deflatenning plaid dataset generation =", end - start)

    # show_cgns_tree(unflat)


    # print("tree deflatenning plaid dataset generation =", end - start)
    # data_dict = huggingface_bridge.reconstruct_flat_tree_from_hf_sample2(hf_dataset_new[0])
    # treedef = deserialize_treedef(data_dict.pop('treedef'))
    # unflat = unflatten_cgns_tree_optree_dict(treedef, data_dict, cgns_types)

    # show_cgns_tree(tree)
    # print("--------------")
    # show_cgns_tree(unflat)

    # print("first sample CGNS trees identical?:", compare_cgns_trees(tree, unflat),)


    # unflat = unflatten_cgns_tree_optree(leaves, treedef, data_dict, cgns_types_dict)
    # print(flat_tree)
    1./0.

    start = time()
    plaid_dataset_new = huggingface_bridge.huggingface_dataset_to_plaid_new2(
        hf_dataset_new, dtypes, cgns_types, processes_number=1
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
