from time import time

import yaml
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from plaid import Dataset, Sample
from plaid.bridges import huggingface_bridge
from plaid.containers.features import SampleFeatures
from plaid.utils.base import get_mem
from plaid.utils.cgns_helper import (
    compare_cgns_trees_no_types,
    show_cgns_tree,
    unflatten_cgns_tree,
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

    # flat, dtypes, cgns_types = flatten_cgns_tree(tree)

    # var_path = [k for k in flat.keys() if (cgns_types[k] in ["DataArray_t", "IndexArray_t", "Zone_t", "Element_t"] and "Time" not in k)]
    # var_flat = {k:v for k,v in flat.items() if k in var_path}
    # cst_flat = {k:v for k,v in flat.items() if k not in var_path}

    # print("-----")
    # print(var_flat.keys())
    # print("-----")
    # print(cst_flat.keys())

    # print("-----")
    # show_cgns_tree(tree)

    # 1./.0

    # keep:
    # DataArray_t
    # IndexArray_t
    # Zone_t

    # no keep:
    # CGNSLibraryVersion_t
    # CGNSBase_t
    # Family_t
    # ZoneType_t
    # GridCoordinates_t

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

    # start = time()
    # hf_dataset_new.set_format("numpy")
    # plaid_dataset_new = huggingface_bridge.huggingface_dataset_to_plaid_new(
    #     hf_dataset_new, dtypes, cgns_types, processes_number=1
    # )
    # end = time()

    # print("tree deflatenning plaid dataset generation =", end - start)

    description = "Converting Hugging Face dataset to plaid"

    sample_list = []
    t1 = t2 = 0
    start = time()
    tic = time()
    # hf_dataset_new.set_format("numpy")
    t0 = time() - tic

    import numpy as np
    import pyarrow as pa

    def iter_rows_fast(dataset):
        """
        Yield rows from a Hugging Face dataset as dicts.
        Tries to convert to NumPy if possible (zero-copy for arrays),
        robust to scalars, None, lists, strings, etc.
        """
        table = dataset.data
        for i in range(len(dataset)):
            row = {}
            for name in table.column_names:
                val = table[name][
                    i
                ]  # could be Scalar, ListScalar, StructScalar, Array, ChunkedArray

                # None
                if val is None or isinstance(val, pa.NullScalar):
                    row[name] = None
                    continue

                # Array or ChunkedArray -> NumPy
                if isinstance(val, (pa.Array, pa.ChunkedArray)):
                    try:
                        arr = val.to_numpy(zero_copy_only=True)
                    except pa.ArrowInvalid:
                        arr = val.to_numpy()
                    row[name] = arr
                    continue

                # Scalar (number, string, list, struct)
                if isinstance(val, pa.Scalar):
                    py_val = val.as_py()
                    # convert single-element list to np.array if you want
                    if isinstance(py_val, list):
                        row[name] = np.asarray(py_val)
                    else:
                        row[name] = py_val
                    continue

                # Fallback: anything else
                row[name] = val

            yield row

    sample_list = []
    for row in tqdm(
        iter_rows_fast(hf_dataset_new), total=len(hf_dataset_new), desc=description
    ):
        unflat = unflatten_cgns_tree(row, 1.0, cgns_types)
        sample_list.append(Sample(features=SampleFeatures({0.0: unflat})))
    plaid_dataset_new = Dataset(samples=sample_list)

    # for idx in tqdm(range(len(hf_dataset_new)), desc=description):
    # # for idx in tqdm(range(10), desc=description):
    #     tic = time()
    #     sample = hf_dataset_new[idx]
    #     t1 += time()-tic
    #     tic = time()
    #     unflat = unflatten_cgns_tree(sample, 1., cgns_types)
    #     t2 += time()-tic
    #     sample_list.append(Sample(features=SampleFeatures({0.0: unflat})))
    # plaid_dataset_new = Dataset(samples=sample_list)
    end = time()
    # print("indiv times =", t0, t1, t2)
    print("tree deflatenning plaid dataset generation =", end - start)

    # for row in iter_rows_fast(hf_dataset_new):
    #     # print(row["Base_2_2/Zone/PointData/U1"][:5])
    #     unflat = unflatten_cgns_tree(row, 1.0, cgns_types)

    #     #print(list(row.keys()))

    # # print(plaid_dataset_new[0].get_field("sig11"))
    # # print(type(plaid_dataset_new[0].get_field("sig11")))

    print(
        "first sample CGNS trees identical (no types)?:",
        compare_cgns_trees_no_types(
            plaid_dataset[0].features.data[0], plaid_dataset_new[0].features.data[0]
        ),
    )

    show_cgns_tree(plaid_dataset[0].features.data[0])
    print("--------------")
    show_cgns_tree(plaid_dataset_new[0].features.data[0])

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
