---
title: Storage backends
---

# Storage tutorial

This tutorial demonstrates end‑to‑end workflows for creating, saving, and loading PLAID datasets using the three storage backends implemented in PLAID:
- hf_datasets (Hugging Face Datasets),
- cgns (plain CGNS files),
- zarr (chunked array storage).

The examples show:
- Preparing dataset metadata (infos) and a ProblemDefinition (features, task, splits).
- Building samples from raw data (ShapeNet‑Car in the example) and yielding them via generator functions.
- Two generation modes: PARALLEL (multiprocess, sharded) and SEQUENTIAL.
- Writing datasets locally with save_to_disk for each backend, and pushing to the Hugging Face Hub with push_to_hub.
- Downloading and instantiating datasets back into PLAID with download_from_hub, init_from_disk and streaming with init_streaming_from_hub.
- Converting raw backend samples to Plaid Sample objects via backend converters and using them efficiently in a PyTorch DataLoader.

Notes and requirements:
- The example uses external tools (plyfile, Muscat) to construct meshes; these are not part of PLAID runtime dependencies.
- Configure a dedicated HF cache (datasets.config.HF_DATASETS_CACHE) when running the Hugging Face backend.
- Parallel mode demonstrates sharding and num_proc controls; writer_batch_size / num_workers are used when uploading to the hub.
- The tutorial includes utilities to inspect CGNS trees and to save single Plaid samples to disk for visualization (e.g., Paraview).

Use these examples as templates to:
- Adapt generators to your raw data format,
- Choose the backend that fits your workflow (hf_datasets for hub integration, cgns for native CGNS interchange, zarr for efficient chunked numeric storage),
- Tune parallelism and sharding to match dataset size and available compute.


## How to create data and save to disk/push to hub

```python
import time
from pathlib import Path
import shutil
from functools import partial

import numpy as np

from datasets import config
from plaid import Sample, ProblemDefinition
from plaid.storage import save_to_disk, push_to_hub

# plyfile and Muscat not included in plaid run dependencies
from plyfile import PlyData
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools.MeshCreationTools import CreateMeshOf
import Muscat.MeshContainers.ElementsDescription as ED


# Use a dedicated temporary cache folder
tmp_cache_dir = "hf_tmp_cache"
config.HF_DATASETS_CACHE = tmp_cache_dir


# Choose to execute the parallel of sequential version
VERSION = "PARALLEL"
N_PROC = 6 # number of parallel processes
N_SHARD = 6 # number of shards, used by the generator in parallel mode, will be overridden by internal logic at writing stage
# VERSION = "SEQUENTIAL"

# raw data dowloaded from https://zenodo.org/records/13993629
# set the folder where the raw data has been downloaded:
BASE_RAW_DATA_FOLDER = "/path/to/raw" # TO UPDATE
# set the folder where the data converted to plaid will be saved locally
BASE_GENERATED_DATA_FOLDER = "/path/to/generated" # TO UPDATE
# set the Huggging Face's repo_id where the datasets will be uploaded
BASE_REPO_ID = "channel/ShapeNetCar" # TO UPDATE


all_backends = ["hf_datasets", "cgns", "zarr"]

#---------------------------------------------------------------
# define some functions to handle ShapeNetCar data

def split_list(lst, n_splits):
    n = len(lst)
    k, m = divmod(n, n_splits)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_splits)]

with open(f"{BASE_RAW_DATA_FOLDER}/train.txt") as f:
    line = f.readline().strip()
    train_ids = [int(x) for x in line.split(",")]

with open(f"{BASE_RAW_DATA_FOLDER}/test.txt") as f:
    line = f.readline().strip()
    test_ids = [int(x) for x in line.split(",")]


base_dir = Path(f"{BASE_RAW_DATA_FOLDER}/data/")

tri_folders = [p for p in base_dir.iterdir() if p.is_dir()]

curated_train_ids = []
curated_test_ids = []

count = 0
for folder in tri_folders:
    id_ = int(folder.name)
    if id_ in train_ids:
        curated_train_ids.append(count)
    else:
        curated_test_ids.append(count)
    count+=1

# we can reduced the number of samples in each split for faster execution
curated_train_ids = curated_train_ids[:10]
curated_test_ids = curated_test_ids[:10]

#---------------------------------------------------------------
# infos and problem definition must be define to correctly populate the dataset's metadata

infos = {"legal": {"owner": "NeuralOperator (https://zenodo.org/records/13993629)", "license": "cc-by-4.0"},
        "data_production": {"physics": "CFD", "type": "simulation",
                            "script": "Converted to PLAID format for standardized access; no changes to data content."},
    }

constant_features = [
"Base_2_3/Zone/Elements_TRI_3/ElementRange",
]

input_features = [
"Base_2_3/Zone/Elements_TRI_3/ElementConnectivity",
"Base_2_3/Zone/GridCoordinates/CoordinateX",
"Base_2_3/Zone/GridCoordinates/CoordinateY",
"Base_2_3/Zone/GridCoordinates/CoordinateZ",
]

output_features = [
"Base_2_3/Zone/VertexFields/pressure",
]


pb_def = ProblemDefinition()
pb_def.add_in_features_identifiers(input_features)
pb_def.add_out_features_identifiers(output_features)
pb_def.add_constant_features_identifiers(constant_features)
pb_def.set_task("regression")
pb_def.set_name("regression_1")
pb_def.set_score_function("RRMSE")
pb_def.set_train_split({"train":"all"})
pb_def.set_test_split({"test":"all"})

#---------------------------------------------------------------

if VERSION == "PARALLEL":

    def _generator(shards_ids):
        for ids in shards_ids:
            for i in ids:
                folder = tri_folders[i]

                plydata = PlyData.read(folder / "tri_mesh.ply")
                tris = np.ascontiguousarray(np.stack(plydata['face'].data['vertex_indices']))

                vertex_data = plydata['vertex'].data
                x = vertex_data['x']
                y = vertex_data['y']
                z = vertex_data['z']

                nodes = np.ascontiguousarray(np.stack((x, y, z)).T)

                mesh = CreateMeshOf(nodes, tris, elemName=ED.Triangle_3)

                press = np.load(folder / "press.npy")
                offset = np.abs(press.shape[0]-mesh.nodes.shape[0])
                mesh.nodeFields["pressure"] = press[offset:]

                tree = MeshToCGNS(mesh, exportOriginalIDs=False)

                sample = Sample()
                sample.add_tree(tree)

                yield sample


    gen_kwargs = {"train": {'shards_ids': split_list(curated_train_ids, N_SHARD)},
                "test": {'shards_ids': split_list(curated_test_ids, N_SHARD)}}

    generators = {"train": _generator,
                "test": _generator}


    for backend in all_backends:

        print("--------------------------------------")
        print(f"Backend: {backend}, parallel version")

        repo_id = f"{BASE_REPO_ID}_{backend}"
        local_folder = f"{BASE_GENERATED_DATA_FOLDER}/{backend}_dataset"

        # DISK
        start = time.time()
        save_to_disk(output_folder = local_folder,
                    generators = generators,
                    backend = backend,
                    infos = infos,
                    pb_defs = pb_def,
                    gen_kwargs = gen_kwargs,
                    num_proc = N_PROC,
                    overwrite = True,
                    verbose = True)
        print(f"duration generate with N_PROC={N_PROC} is {time.time()-start} s")

        # HUB
        start = time.time()
        push_to_hub(repo_id = repo_id,
                    local_dir = local_folder,
                    num_workers = N_PROC,
                    viewer = backend == "hf_datasets",
                    illustration_urls=["https://i.ibb.co/3mGHsHMk/Shape-Net-Car-samples.png"])
        print(f"duration push to hub N_PROC={N_PROC} is {time.time()-start} s")

#---------------------------------------------------------------

if VERSION == "SEQUENTIAL":

    def _generator(ids):
        for i in ids:
            folder = tri_folders[i]

            plydata = PlyData.read(folder / "tri_mesh.ply")
            tris = np.ascontiguousarray(np.stack(plydata['face'].data['vertex_indices']))

            vertex_data = plydata['vertex'].data
            x = vertex_data['x']
            y = vertex_data['y']
            z = vertex_data['z']

            nodes = np.ascontiguousarray(np.stack((x, y, z)).T)

            mesh = CreateMeshOf(nodes, tris, elemName=ED.Triangle_3)

            press = np.load(folder / "press.npy")
            offset = np.abs(press.shape[0]-mesh.nodes.shape[0])
            mesh.nodeFields["pressure"] = press[offset:]

            tree = MeshToCGNS(mesh, exportOriginalIDs=False)

            sample = Sample()
            sample.add_tree(tree)

            yield sample

    generators = {"train": partial(_generator, range(len(curated_train_ids))),
                "test": partial(_generator, range(len(curated_test_ids)))}

    for backend in all_backends:

        print("--------------------------------------")
        print(f"Backend: {backend}, sequential version")

        local_folder = f"{BASE_GENERATED_DATA_FOLDER}/{backend}_dataset"
        # DISK
        start = time.time()
        save_to_disk(output_folder=local_folder,
                    generators = generators,
                    backend = backend,
                    infos = infos,
                    pb_defs = pb_def,
                    overwrite=True,
                    verbose=True)

        print(f"duration generate with N_PROC={N_PROC} and N_SHARD={N_SHARD} is {time.time()-start} s")

if Path(tmp_cache_dir).exists():
    shutil.rmtree(Path(tmp_cache_dir))
```

## How to read data from disk/hub

```python
import time

# pytorch not included in plaid dependencies
import torch
from torch.utils.data import DataLoader

from plaid.utils.cgns_helper import show_cgns_tree
from plaid.storage import init_from_disk, download_from_hub, init_streaming_from_hub
from plaid.storage import load_problem_definitions_from_disk


# set the Huggging Face's repo_id from which the datasets will be downloaded
BASE_REPO_ID = "channel/ShapeNetCar" # TO UPDATE
# set the folder where the downloaded data will be saved locally
BASE_DOWNLOADED_DATA_FOLDER = "/mnt/e/converted_datasets/ShapeNet-Car" # TO UPDATE

all_backends = ["hf_datasets", "cgns", "zarr"]
split = "train"

# Load problem definitions and define features as all the input and output features
pb_defs = load_problem_definitions_from_disk(f"{BASE_DOWNLOADED_DATA_FOLDER}/{all_backends[0]}_dataset")
pb_def = pb_defs[0]
features = pb_def.get_in_features_identifiers() + pb_def.get_out_features_identifiers()

print("----------------------------------------------------")
print("-- Download datasets -------------------------------")
print("----------------------------------------------------")

# download datasets
for backend in all_backends:
    BACKEND = all_backends[0]
    repo_id = f"{BASE_REPO_ID}_{BACKEND}"
    download_folder = f"{BASE_DOWNLOADED_DATA_FOLDER}/downloaded_{BACKEND}_dataset"

    # depending on the backends, one can download a subset of the samples and features. We keep them all here
    split_ids_ = None
    features_ = None

    download_from_hub(repo_id, download_folder, split_ids = split_ids_, features = features_, overwrite = True)


print("-------------------------------------------------------")
print("-- Dataset local read and plaid sample instantiation --")
print("-------------------------------------------------------")

for backend in all_backends:

    datasetdict, converterdict = init_from_disk(f"{BASE_DOWNLOADED_DATA_FOLDER}/downloaded_{backend}_dataset")

    # specify one dataset/converter pair for one split
    dataset = datasetdict[split]
    converter = converterdict[split]

    print("backend: ", converter.backend)

    # generic way to instantiate all the samples
    start = time.time()
    for i in range(len(dataset)):
        plaid_sample = converter.to_plaid(dataset, i)
    print(f"duration {time.time()-start}")

    # instantiate the first sample, depends on the backend
    sample = dataset[0]
    # alternative way instantiate a plaid sample (much slower for hf_datasets)
    plaid_sample = converter.sample_to_plaid(dataset[0])

    # save a plaid sample in a CGNS that can be opened in paraview
    plaid_sample.save_to_dir(f"{BASE_DOWNLOADED_DATA_FOLDER}/sample_0_{backend}", overwrite = True)

    # generic way to read all features for all time steps
    for t in plaid_sample.get_all_time_values():
        for path in pb_def.get_in_features_identifiers():
            plaid_sample.get_feature_by_path(path=path, time=t)
        for path in pb_def.get_out_features_identifiers():
            plaid_sample.get_feature_by_path(path=path, time=t)

    # generic way to return the data as a dict containing all constant and variable features
    if backend != "cgns":
        sample_dict = converter.to_dict(dataset, 0)
        sample_dict = converter.sample_to_dict(dataset[0])

    # alternative way to return the data as a dict containing all constant and variable features from a plaid sample
    sample_dict = converter.plaid_to_dict(plaid_sample)

    print("----------")


print("----------------------------------------------------")
print("-- Torch dataloader + send to GPU ------------------")
print("----------------------------------------------------")

# define a simple class for efficient torch Dataloader iterations
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx

for backend in all_backends:
    datasetdict, converterdict = init_from_disk(f"{BASE_DOWNLOADED_DATA_FOLDER}/{backend}_dataset")
    dataset = datasetdict[split]
    converter = converterdict[split]

    # define a torch dataloader directly from this IndexDataset class
    loader = DataLoader(
        IndexDataset(len(dataset)),
        batch_size=10,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )
    print("backend: ", converter.backend)
    start = time.time()
    for batch in loader:
        for idx in batch:
            # efficient plaid sample reconstruction
            plaid_sample = converter.to_plaid(dataset, idx)
            # generic way of retrieving features and send them to GPU
            for time_ in plaid_sample.get_all_mesh_times():
                torch_sample = {}
                for path in features:
                    value = plaid_sample.get_feature_by_path(path=path, time=time_)
                    if value is not None:
                        if not value.flags.writeable:
                            value = value.copy()
                        torch_sample[path] = torch.as_tensor(value).to("cuda", non_blocking=True)
    print(f"duration {time.time()-start}")
    print("----------")


print("----------------------------------------------------")
print("-- Streaming test ----------------------------------")
print("----------------------------------------------------")


for backend in all_backends:

    datasetdict, converterdict = init_streaming_from_hub(f"{BASE_REPO_ID}_{backend}")

    dataset = datasetdict[split]
    converter = converterdict[split]

    # dataset here is an IterableDataset, retrieving one sample and converting it to plaid
    raw_sample = next(iter(dataset))
    plaid_sample = converter.sample_to_plaid(raw_sample)

    # utility to print a summary of the CGNS tree from the plaid sample
    show_cgns_tree(plaid_sample.get_tree(0.))
```