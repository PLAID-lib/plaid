"""
PLAID Dataset Conversion Example

Dataset:
- The Well: Turbulent Radiative Layer 2D

Purpose:
- Convert a dataset with structured grids and temporal trajectories
  to the PLAID format for publication.

Notes:
- Dataset-specific logic and PLAID semantics are intentionally explicit
- External, dataset-specific dependencies may be required
- Script is not meant to be fully generic or reusable as-is
"""


from Muscat.Bridges.CGNSBridge import MeshToCGNS
from plaid.storage import save_to_disk, push_to_hub
from plaid import Sample, ProblemDefinition
import numpy as np
import time
import glob
import h5py

from functools import partial

from Muscat.MeshTools.ConstantRectilinearMeshTools import CreateConstantRectilinearMesh
from Muscat.MeshContainers.Filters.FilterObjects import NodeFilter

N_PROC = 9
N_SHARD = 9

def split_list(lst, n_splits):
    n = len(lst)
    k, m = divmod(n, n_splits)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_splits)]


repo_id = "channel/repo"
local_folder = "/path/to/local/folder"

assert repo_id != "channel/repo", "Please set repo_id"
assert local_folder != "/path/to/local/folder", "Please set local_folder"


mesh = CreateConstantRectilinearMesh(dimensions=[128,384], origin=[-0.5,-1.], spacing=[1/127, 3/383])


nf = NodeFilter(zone=lambda p: (p[:, 0]+0.4999))
indicesx1 = nf.GetNodesIndices(mesh)
nf = NodeFilter(zone=lambda p: (-p[:, 0]+0.4999))
indicesx2 = nf.GetNodesIndices(mesh)

x_periodic = np.hstack((indicesx1, indicesx2))

mesh.GetNodalTag("x_periodic").AddToTag(x_periodic)


nf = NodeFilter(zone=lambda p: (p[:, 1]+0.9999))
indicesy1 = nf.GetNodesIndices(mesh)
nf = NodeFilter(zone=lambda p: (-p[:, 1]+1.9999))
indicesy2 = nf.GetNodesIndices(mesh)

y_open = np.hstack((indicesy1, indicesy2))

mesh.GetNodalTag("y_open").AddToTag(y_open)


keys = ['boundary_conditions', 'dimensions', 'scalars', 't0_fields', 't1_fields', 't2_fields']


splits = ["train", "test", "valid"]

split_file_names = {split:glob.glob(f"/mnt/e/raw_datasets/TheWell/turbulent_radiative_layer_2D/raw/data/{split}/*.hdf5") for split in splits}



infos = {"legal": {"owner": "polymathic-ai (https://huggingface.co/datasets/polymathic-ai/turbulent_radiative_layer_2D)", "license": "cc-by-4.0"},
        "data_production": {"physics": "Turbulent Radiative Layer - 2D", "simulator": "Athena++", "type": "simulation",
                            "script": "Converted to PLAID format for standardized access; no changes to data content."}
    }

input_features = [
"Base_2_2/Zone/VertexFields/v_x_IC",
"Base_2_2/Zone/VertexFields/v_y_IC",
"Global/tcool"
]

output_features = [
"Base_2_2/Zone/VertexFields/density",
"Base_2_2/Zone/VertexFields/pressure",
"Base_2_2/Zone/VertexFields/v_x",
"Base_2_2/Zone/VertexFields/v_y",
]

constant_features = [
"Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
"Base_2_2/Zone/Elements_QUAD_4/ElementRange",
"Base_2_2/Zone/GridCoordinates/CoordinateX",
"Base_2_2/Zone/GridCoordinates/CoordinateY",
"Base_2_2/Zone/VertexFields/density_IC",
"Base_2_2/Zone/VertexFields/pressure_IC",
"Base_2_2/Zone/ZoneBC/x_periodic/PointList",
"Base_2_2/Zone/ZoneBC/y_open/PointList",
]


pb_def = ProblemDefinition()
pb_def.add_in_features_identifiers(input_features)
pb_def.add_out_features_identifiers(output_features)
pb_def.add_constant_features_identifiers(constant_features)
pb_def.set_task("regression")
pb_def.set_name("regression")
pb_def.set_score_function("RRMSE")
pb_def.set_train_split({"train":"all", "valid":"all"})
pb_def.set_test_split({"test":"all"})

#----------------------------------------------------------------------

def _generator(shards_ids, split):
    for ids in shards_ids:
        for i in ids:
            file_name = split_file_names[split][i]
            with h5py.File(file_name, "r") as f:


                time = f["/dimensions/time"][()]

                tcool = f["/scalars/tcool"][()]

                density = f["/t0_fields/density"][()]
                pressure = f["/t0_fields/pressure"][()]

                velocity = f["/t1_fields/velocity"][()]

                n_traj = density.shape[0]

                for i_traj in range(n_traj):

                    sample = Sample()

                    for i, t in enumerate(time):
                        tree = MeshToCGNS(mesh, exportOriginalIDs=False)
                        sample.add_tree(tree, time=t)

                        if i==0:
                            sample.add_field("density_IC", density[i_traj,i,:,:].flatten(), time=t)
                            sample.add_field("pressure_IC", pressure[i_traj,i,:,:].flatten(), time=t)
                            sample.add_field("v_x_IC", velocity[i_traj,i,:,:,0].flatten(), time=t)
                            sample.add_field("v_y_IC", velocity[i_traj,i,:,:,1].flatten(), time=t)

                        sample.add_field("density", density[i_traj,i,:,:].flatten(), time=t)
                        sample.add_field("pressure", pressure[i_traj,i,:,:].flatten(), time=t)
                        sample.add_field("v_x", velocity[i_traj,i,:,:,0].flatten(), time=t)
                        sample.add_field("v_y", velocity[i_traj,i,:,:,1].flatten(), time=t)

                    sample.add_scalar("tcool", tcool)

                    yield sample

generators = {split:partial(_generator, split=split) for split in splits}
gen_kwargs = {split: {'shards_ids': split_list(np.arange(len(split_file_names[split])), N_SHARD)} for split in splits}

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
            pretty_name = "TheWell_turbulent_radiative_layer_2D",
            illustration_urls = ["https://i.ibb.co/2XvrHwX/turb-rad-layer-2d-samples.gif"])
print(f"duration push to hub N_PROC={N_PROC} is {time.time()-start} s")