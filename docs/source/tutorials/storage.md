---
title: Storage backends
---

# Storage tutorial

End‑to‑end workflows for creating, saving, and loading PLAID datasets with the three storage backends: **hf_datasets**, **cgns**, and **zarr**.

## Key concepts

- **`sample_constructor`** is a simple function that takes a single identifier (of any type) and returns a PLAID `Sample`. The identifier can be an integer, a file path, a string, a tuple — anything that makes sense for your data.
- **`ids`** is a dictionary mapping split names to **sliceable sequences** of identifiers — anything with `__getitem__` and `__len__` (list, tuple, numpy array, …). PLAID handles iteration, generator creation, and parallel sharding internally.
- **`save_to_disk`** writes a dataset locally; **`push_to_hub`** uploads it to Hugging Face Hub.
- **`init_from_disk`** / **`download_from_hub`** / **`init_streaming_from_hub`** load datasets back into PLAID.
- Backend converters turn raw backend samples into PLAID `Sample` objects.

## Notes

- The example uses external tools (`plyfile`, `Muscat`) to build meshes — these are not PLAID runtime dependencies.
- Set `datasets.config.HF_DATASETS_CACHE` to a dedicated folder when using the HF backend.
- Loading metadata from **local disk** keeps numeric constants as `np.memmap` for memory efficiency; loading from the **Hub** materializes them into in-memory arrays to avoid lifetime issues with temporary download folders.


## How to create data and save to disk/push to hub

```python
import time
from pathlib import Path
import shutil

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


N_PROC = 6 # number of parallel processes (set to 1 for sequential execution)

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
# Define a simple function that takes a single identifier and returns a Sample.
# PLAID handles iteration, generator creation, and parallel sharding internally.
# When num_proc > 1, PLAID automatically shards the ids across workers.

def sample_constructor(i):
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

    return sample

ids = {"train": curated_train_ids,
       "test": curated_test_ids}

for backend in all_backends:

    print("--------------------------------------")
    print(f"Backend: {backend}, N_PROC: {N_PROC}")

    repo_id = f"{BASE_REPO_ID}_{backend}"
    local_folder = f"{BASE_GENERATED_DATA_FOLDER}/{backend}_dataset"

    # DISK
    start = time.time()
    save_to_disk(output_folder=local_folder,
                sample_constructor=sample_constructor,
                ids=ids,
                backend=backend,
                infos=infos,
                pb_defs=pb_def,
                num_proc=N_PROC,
                overwrite=True,
                verbose=True)
    print(f"duration generate with num_proc={N_PROC} is {time.time()-start} s")

    # HUB
    start = time.time()
    push_to_hub(repo_id=repo_id,
                local_dir=local_folder,
                num_workers=N_PROC,
                viewer=backend == "hf_datasets",
                illustration_urls=["https://i.ibb.co/3mGHsHMk/Shape-Net-Car-samples.png"])
    print(f"duration push to hub N_PROC={N_PROC} is {time.time()-start} s")

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
    repo_id = f"{BASE_REPO_ID}_{backend}"
    download_folder = f"{BASE_DOWNLOADED_DATA_FOLDER}/downloaded_{backend}_dataset"

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

    # Optional: extract only selected indices inside specific variable features
    # (currently supported for hf_datasets and zarr backends).
    field_path = "Base_2_3/Zone/VertexFields/pressure"
    selected_idx = [0, 10, 20, 30]
    plaid_sample_sub = converter.to_plaid(
        dataset,
        0,
        features=[field_path],
        indexers={field_path: selected_idx},
    )

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

```

### Indexed extraction with `indexers`

`converter.to_dict(...)` and `converter.to_plaid(...)` accept an optional
`indexers` argument:

```python
sample = converter.to_plaid(
    dataset,
    idx=0,
    features=["Base/Zone/VertexFields/mach"],
    indexers={"Base/Zone/VertexFields/mach": [1, 5, 9]},
)
```

- `indexers` is a mapping `feature_path -> indexer` (list/array of indices or slice).
- Indexing is applied on the **last axis** of each indexed feature.
- This enables a “read less + one gathered output copy” behavior:
  - **zarr**: partial chunk reads + gathered output
  - **hf_datasets**: Arrow/NumPy best-effort gather + gathered output
- `cgns` backend does not use this mechanism.


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
