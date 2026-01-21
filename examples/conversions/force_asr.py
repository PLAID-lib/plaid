"""
PLAID Dataset Conversion Example

Dataset:
- ForceASR (phase-field fracture simulations, subset 'res-SENS')

Purpose:
- Convert a time-dependent simulation dataset with unstructured meshes,
  external time metadata, and mixed global and nodal quantities
  to the PLAID format for publication.

Notes:
- Each Sample represents one full temporal simulation
- Time steps are extracted from external metadata (PVD files)
- Mesh geometry is unstructured and varies across samples
- Both nodal fields and global quantities are time-dependent
- External, dataset-specific dependencies are required (e.g. Muscat, VTK)
- Script performs temporal subsampling and is not meant to be generic
"""

from Muscat.IO.VtkReader import VtkReader
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from plaid.storage import save_to_disk, push_to_hub
from plaid import Sample, ProblemDefinition
import numpy as np
from pathlib import Path
from tqdm import tqdm
import xml.etree.ElementTree as ET
import os, re
import time


# downloaded from https://zenodo.org/records/7445749

N_PROC = 14
N_SHARD = 14

skip_times = 5


def split_list(lst, n_splits):
    n = len(lst)
    k, m = divmod(n, n_splits)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_splits)]


repo_id = "channel/repo"
local_folder = "/path/to/local/folder"

assert repo_id != "channel/repo", "Please set repo_id"
assert local_folder != "/path/to/local/folder", "Please set local_folder"


def extract_time_sequence(pvd_path):
    tree = ET.parse(pvd_path)
    root = tree.getroot()

    times = []
    for dataset in root.findall(".//DataSet"):
        time_value = dataset.get("timestep")
        if time_value is not None:
            times.append(float(time_value))
    return times



def extract_column_names(path):
    with open(path, 'r') as f:
        header = f.readline().rstrip("\n")
        second = f.readline().rstrip("\n")

    sep_matches = list(re.finditer(r'\s{2,}', second))

    split_positions = [m.start() for m in sep_matches]

    def split_at_positions(s, positions):
        parts = []
        last = 0
        for pos in positions:
            parts.append(s[last:pos])
            last = pos
        parts.append(s[last:])
        return [p.strip() for p in parts]

    return split_at_positions(header, split_positions)

def leaf_dirs(root):
    """Yield all leaf (lowest-level) directories under root."""
    has_subdir = False
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                has_subdir = True
                yield from leaf_dirs(entry.path)
    if not has_subdir:
        yield Path(root)

root = Path("/mnt/e/raw_datasets/ForceASR/raw/res-SENS")
all_directories = leaf_dirs(root)

directories = []

for base_dir in all_directories:

    assert base_dir.parts[6] == "res-SENS"

    if "tensor" in str(base_dir):
        continue

    fileNames = list(base_dir.rglob("*.vtu"))
    if len(fileNames) == 0:
        continue

    directories.append(base_dir)


#----------------------------------------------------------------------
infos = {"legal": {"owner": "RK 2423 FRASCAL (https://zenodo.org/records/7445749)", "license": "cc-by-4.0"},
        "data_production": {"physics": "phase-field fracture models for brittle fracture", "type": "simulation",
                            "script": "Subset 'res-SENS' of the initial dataset, 1/5th time steps, converted to PLAID format for standardized access; no changes to data content."},
    }


constant_features = [
    "Global/displacement",
    "Global/fracture energy",
    "Global/strain energy",
    "Global/total energy",
]

input_features = [
    "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
    "Base_2_2/Zone/Elements_QUAD_4/ElementRange",
    "Base_2_2/Zone/GridCoordinates/CoordinateX",
    "Base_2_2/Zone/GridCoordinates/CoordinateY",
    "Base_2_2/Zone/VertexFields/materialID",
    "Global/config",
    "Global/fref",
    "Global/pfThres",
    "Global/x-force",
    "Global/y-force",
]

output_features = [
    "Base_2_2/Zone/VertexFields/Displacement_X",
    "Base_2_2/Zone/VertexFields/Displacement_Y",
    "Base_2_2/Zone/VertexFields/PhaseField",
]

pb_def = ProblemDefinition()
pb_def.add_in_features_identifiers(input_features)
pb_def.add_out_features_identifiers(output_features)
pb_def.add_constant_features_identifiers(constant_features)
pb_def.set_task("regression")
pb_def.set_name("regression")
pb_def.set_score_function("RRMSE")
pb_def.set_train_split({"train":"all"})
pb_def.set_test_split({"test":"all"})


#----------------------------------------------------------------------


reader = VtkReader()

def _generator(shards_ids):

    for ids in shards_ids:

        for ind in ids:

            base_dir = directories[ind]

            assert base_dir.parts[6] == "res-SENS"

            if "tensor" in str(base_dir):
                continue

            fileNames = list(base_dir.rglob("*.vtu"))
            if len(fileNames) == 0:
                continue

            rank_files = np.argsort([int(str(fn).split('-')[-1][:-4]) for fn in fileNames])

            fileNames = [fileNames[r] for r in rank_files][::skip_times]

            pvd_file = list(base_dir.rglob("*.pvd"))[0]
            times = extract_time_sequence(pvd_file)
            times = times[::skip_times]

            dat_file = list(base_dir.rglob("*.dat"))[0]

            globals = np.loadtxt(dat_file, skiprows=1)
            globals = globals[skip_times-1::skip_times,:]

            global_names = extract_column_names(dat_file)

            sample = Sample()

            sample.add_global("config", str(base_dir)[42:])

            match = re.search(r"fref(\.\d+)", base_dir.name)
            n_match = 0
            if match is not None:
                fref = float(match.group(1))
                sample.add_global("fref", fref)
                n_match += 1

            match = re.search(r"pfThres(\.\d+)", base_dir.name)
            if match is not None:
                pfThres = float(match.group(1))
                sample.add_global("pfThres", pfThres)
                n_match += 1

            assert n_match<2

            i = 0
            for fileName in tqdm(fileNames):

                reader.SetFileName(fileName)
                mesh = reader.Read()
                mesh.nodes = np.ascontiguousarray(mesh.nodes[:,:2])

                displacement = np.ascontiguousarray(mesh.nodeFields["Displacement"][:,:2])
                phasefield = np.ascontiguousarray(mesh.nodeFields["PhaseField"])
                materialID = np.ascontiguousarray(mesh.nodeFields["materialID"])

                mesh.nodeFields = {}

                tree = MeshToCGNS(mesh, exportOriginalIDs=False)

                sample.add_tree(tree, time=times[i])
                sample.add_field("Displacement_X", displacement[:,0], time=times[i])
                sample.add_field("Displacement_Y", displacement[:,1], time=times[i])
                sample.add_field("PhaseField", phasefield, time=times[i])
                sample.add_field("materialID", materialID, time=times[i])

                if i>0:
                    assert globals[i-1,0] == times[i]
                    for j, gn in enumerate(global_names[1:]):
                        sample.features.add_global(gn, globals[i-1,j+1], time = times[i])
                else:
                    for j, gn in enumerate(global_names[1:]):
                        sample.features.add_global(gn, 0., time = times[i])

                i += 1


            yield sample

gen_kwargs = {"res_SENS": {'shards_ids': split_list(np.arange(len(directories)), N_SHARD)},}
generators = {"res_SENS": _generator}


# DISK
start = time.time()
save_to_disk(output_folder = local_folder,
            generators = generators,
            backend = "hf_datasets",
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
            viewer = True,
            pretty_name = "ForceASR dataset",
            illustration_urls=["https://i.ibb.co/gZtL8VrY/force-ASR-samples.gif"])
print(f"duration push to hub N_PROC={N_PROC} is {time.time()-start} s")

