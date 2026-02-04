"""
PLAID Dataset Conversion Example

Dataset:
- ShapeNet-Car (triangular surface meshes with scalar fields)

Purpose:
- Convert a dataset of unstructured triangular meshes and associated
  nodal fields to the PLAID format, and publish it using multiple backends.

Notes:
- Each Sample represents a single static geometry (no temporal dimension)
- Mesh topology and geometry are dataset-specific and constructed explicitly
- External, dataset-specific dependencies are required (e.g. plyfile, Muscat)
- Script demonstrates both sequential and parallel generator patterns
- Script is not meant to be generic or reusable as-is
"""

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
BASE_RAW_DATA_FOLDER = "/path/to/raw"
# set the folder where the data converted to plaid will be saved locally
BASE_GENERATED_DATA_FOLDER = "/path/to/local/folder"
# set the Huggging Face's repo_id where the datasets will be uploaded
BASE_REPO_ID = "channel/repo"

assert BASE_REPO_ID != "channel/repo", "Please set BASE_REPO_ID"
assert BASE_GENERATED_DATA_FOLDER != "/path/to/local/folder", "Please set BASE_GENERATED_DATA_FOLDER"
assert BASE_RAW_DATA_FOLDER != "/path/to/raw", "Please set BASE_RAW_DATA_FOLDER"

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