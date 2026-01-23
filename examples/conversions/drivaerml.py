"""
PLAID Dataset Conversion Example

Dataset:
- DrivAerML: High-Fidelity CFD Dataset for Road-Car External Aerodynamics

Purpose:
- Convert OpenFOAM steady-state CFD simulations of automotive aerodynamics
  to the PLAID format.

Notes:
- Each Sample represents one static CFD simulation (steady-state, no time dimension)
- OpenFOAM polyMesh format with unstructured hexahedral/polyhedral cells
- Fields include velocity (U), pressure (p), and modified pressure (p_rgh)
- Geometric parameters and force/moment coefficients available in CSV files
- External dependencies required (e.g., pandas, Muscat or custom OpenFOAM parser)
- Script is not meant to be generic or reusable as-is
"""

import time
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

from plaid import Sample, ProblemDefinition
from plaid.storage import save_to_disk, push_to_hub

# External dependencies for OpenFOAM parsing
# NOTE: This example assumes availability of OpenFOAM parsing utilities
# Users may need to adapt based on their preferred OpenFOAM Python interface
# Options include: PyFoam, Muscat with OpenFOAM support, or custom parsers

# For this example, we'll outline the structure assuming a custom parser
# that reads OpenFOAM polyMesh and field files


# Configuration
VERSION = "SEQUENTIAL"  # or "PARALLEL"
N_PROC = 6  # for parallel version

# Paths - MUST BE CONFIGURED BY USER
BASE_RAW_DATA_FOLDER = "/path/to/drivaerml/download"
BASE_GENERATED_DATA_FOLDER = "/path/to/local/output"
BASE_REPO_ID = "channel/drivaerml"

assert BASE_REPO_ID != "channel/drivaerml", "Please set BASE_REPO_ID"
assert BASE_GENERATED_DATA_FOLDER != "/path/to/local/output", "Please set BASE_GENERATED_DATA_FOLDER"
assert BASE_RAW_DATA_FOLDER != "/path/to/drivaerml/download", "Please set BASE_RAW_DATA_FOLDER"

# Backends to use
all_backends = ["hf_datasets", "cgns", "zarr"]


#---------------------------------------------------------------
# OpenFOAM parsing utilities (placeholder - users must implement)
#---------------------------------------------------------------

def read_openfoam_polymesh(case_dir):
    """
    Read OpenFOAM polyMesh structure from constant/polyMesh/ directory.
    
    Returns:
        mesh: A mesh object compatible with PLAID (e.g., Muscat mesh)
    
    NOTE: This is a placeholder. Users must implement or use existing tools:
    - PyFoam: from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
    - Custom parser for points, faces, owner, neighbour, boundary files
    - Muscat integration if available for OpenFOAM
    """
    mesh_dir = Path(case_dir) / "constant" / "polyMesh"
    
    # Read points file
    points_file = mesh_dir / "points"
    # TODO: Parse OpenFOAM ASCII format for point coordinates
    
    # Read faces file
    faces_file = mesh_dir / "faces"
    # TODO: Parse OpenFOAM face definitions
    
    # Read owner and neighbour files
    owner_file = mesh_dir / "owner"
    neighbour_file = mesh_dir / "neighbour"
    # TODO: Parse cell-face connectivity
    
    # Read boundary file
    boundary_file = mesh_dir / "boundary"
    # TODO: Parse boundary patches
    
    raise NotImplementedError(
        "OpenFOAM polyMesh parsing must be implemented. "
        "Users should use PyFoam, custom parsers, or Muscat utilities."
    )
    
    # Expected return: mesh object with nodes, elements, and structure


def read_openfoam_field(case_dir, field_name, time_dir="0"):
    """
    Read OpenFOAM field data from time directory.
    
    Args:
        case_dir: Path to OpenFOAM case
        field_name: Name of field (e.g., 'U', 'p', 'p_rgh')
        time_dir: Time directory name (default "0" for initial/steady state)
    
    Returns:
        field_data: numpy array with field values
    
    NOTE: This is a placeholder. Users must implement field parsing.
    """
    field_file = Path(case_dir) / time_dir / field_name
    
    # TODO: Parse OpenFOAM field file
    # - Handle internalField (cell values or node values)
    # - Handle boundaryField if needed
    # - Extract dimensions and data
    
    raise NotImplementedError(
        "OpenFOAM field parsing must be implemented. "
        "Users should use PyFoam or custom parsers."
    )
    
    # Expected return: numpy array of field values


def openfoam_to_cgns_tree(case_dir):
    """
    Convert OpenFOAM case to CGNS tree structure compatible with PLAID.
    
    This function should:
    1. Read polyMesh structure
    2. Read field data (U, p, p_rgh)
    3. Convert to CGNS tree using Muscat or similar
    4. Attach fields appropriately (nodal or cell-centered)
    
    Returns:
        tree: CGNS tree object
    """
    # Read mesh
    mesh = read_openfoam_polymesh(case_dir)
    
    # Read fields
    velocity = read_openfoam_field(case_dir, "U", "0")
    pressure = read_openfoam_field(case_dir, "p", "0")
    pressure_rgh = read_openfoam_field(case_dir, "p_rgh", "0")
    
    # Attach fields to mesh
    # Determine if fields are cell-centered or nodal from OpenFOAM internalField
    # OpenFOAM typically uses cell-centered fields
    mesh.elemFields = {}
    mesh.elemFields["U_x"] = velocity[:, 0]
    mesh.elemFields["U_y"] = velocity[:, 1]
    mesh.elemFields["U_z"] = velocity[:, 2]
    mesh.elemFields["p"] = pressure
    mesh.elemFields["p_rgh"] = pressure_rgh
    
    # Convert to CGNS tree
    from Muscat.Bridges.CGNSBridge import MeshToCGNS
    tree = MeshToCGNS(mesh, exportOriginalIDs=False)
    
    return tree


#---------------------------------------------------------------
# Dataset structure and metadata
#---------------------------------------------------------------

base_dir = Path(BASE_RAW_DATA_FOLDER)

# Load CSV files with metadata
geo_params_df = pd.read_csv(base_dir / "geo_parameters_all.csv")
forces_df = pd.read_csv(base_dir / "force_mom_all.csv")
forces_constref_df = pd.read_csv(base_dir / "force_mom_constref_all.csv")

# Identify all run directories
# Pattern: run_0, run_000, run_001, ..., run_258
run_dirs = []
for pattern in ["run_0"] + [f"run_{i:03d}" for i in range(259)]:
    run_path = base_dir / "openfoam_meshes" / pattern
    if run_path.exists():
        run_dirs.append(run_path)

# For demonstration, limit to first few runs
# Remove this line to process all runs
run_dirs = run_dirs[:10]

# Create train/test split
# Users should define appropriate split based on their needs
n_samples = len(run_dirs)
train_ids = list(range(int(0.8 * n_samples)))
test_ids = list(range(int(0.8 * n_samples), n_samples))


#---------------------------------------------------------------
# Problem definition and metadata
#---------------------------------------------------------------

infos = {
    "legal": {
        "owner": "Neil Ashton (https://caemldatasets.org, https://huggingface.co/datasets/neashton/drivaerml)",
        "license": "cc-by-sa-4.0"
    },
    "data_production": {
        "physics": "CFD - Automotive External Aerodynamics",
        "type": "simulation",
        "script": "Converted to PLAID format for standardized access; no changes to data content.",
    },
}

# Define features based on OpenFOAM fields and CSV data
# NOTE: Adjust these based on actual CGNS tree structure after conversion
constant_features = [
    "Base_X_X/Zone/Elements_XXXX/ElementConnectivity",
    "Base_X_X/Zone/Elements_XXXX/ElementRange",
    "Base_X_X/Zone/GridCoordinates/CoordinateX",
    "Base_X_X/Zone/GridCoordinates/CoordinateY",
    "Base_X_X/Zone/GridCoordinates/CoordinateZ",
]

input_features = [
    # Velocity components (cell-centered or nodal)
    "Base_X_X/Zone/CellData/U_x",  # or VertexFields depending on OpenFOAM field location
    "Base_X_X/Zone/CellData/U_y",
    "Base_X_X/Zone/CellData/U_z",
    # Pressure fields
    "Base_X_X/Zone/CellData/p",
    "Base_X_X/Zone/CellData/p_rgh",
]

output_features = [
    # Force and moment coefficients from CSV
    "Global/C_D",  # Drag coefficient
    "Global/C_L",  # Lift coefficient
    "Global/C_S",  # Side force coefficient (if available)
    # Add other force/moment coefficients as needed
]

pb_def = ProblemDefinition()
pb_def.add_in_features_identifiers(input_features)
pb_def.add_out_features_identifiers(output_features)
pb_def.add_constant_features_identifiers(constant_features)
pb_def.set_task("regression")
pb_def.set_name("aerodynamic_forces_prediction")
pb_def.set_score_function("RRMSE")
pb_def.set_train_split({"train": train_ids})
pb_def.set_test_split({"test": test_ids})


#---------------------------------------------------------------
# Generator functions
#---------------------------------------------------------------

def _generator(ids):
    """
    Generator that yields PLAID Samples from OpenFOAM cases.
    
    Args:
        ids: List of indices into run_dirs
    """
    for i in ids:
        case_dir = run_dirs[i]
        
        # Convert OpenFOAM case to CGNS tree
        tree = openfoam_to_cgns_tree(case_dir)
        
        # Create Sample
        sample = Sample()
        sample.add_tree(tree)
        
        # Add global quantities from CSV files
        # Match run directory name to CSV row
        # NOTE: Users must implement proper matching logic
        run_name = case_dir.name
        # Example: extract run number and match to CSV
        # run_number = int(run_name.split('_')[-1])
        # row = forces_df.iloc[run_number]
        # sample.add_scalar(name="C_D", value=row["C_D"])
        # sample.add_scalar(name="C_L", value=row["C_L"])
        
        yield sample


#---------------------------------------------------------------
# Sequential version
#---------------------------------------------------------------

if VERSION == "SEQUENTIAL":
    
    generators = {
        "train": partial(_generator, train_ids),
        "test": partial(_generator, test_ids)
    }
    
    for backend in all_backends:
        print("--------------------------------------")
        print(f"Backend: {backend}, sequential version")
        
        local_folder = f"{BASE_GENERATED_DATA_FOLDER}/drivaerml_{backend}"
        
        start = time.time()
        save_to_disk(
            output_folder=local_folder,
            generators=generators,
            backend=backend,
            infos=infos,
            pb_defs=pb_def,
            overwrite=True,
            verbose=True
        )
        print(f"Duration: {time.time() - start:.2f} s")
        
        # Push to HuggingFace Hub
        if backend == "hf_datasets":
            push_to_hub(
                repo_id=f"{BASE_REPO_ID}_{backend}",
                local_dir=local_folder,
                viewer=True,
                arxiv_paper_urls=["https://arxiv.org/abs/2408.11969"],
                pretty_name="DrivAerML - Road Car External Aerodynamics CFD Dataset"
            )


#---------------------------------------------------------------
# Parallel version (optional)
#---------------------------------------------------------------

if VERSION == "PARALLEL":
    
    def split_list(lst, n_splits):
        """Split list into n_splits roughly equal parts."""
        n = len(lst)
        k, m = divmod(n, n_splits)
        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n_splits)]
    
    def _generator_parallel(shards_ids):
        """Parallel generator that processes multiple shards."""
        for ids in shards_ids:
            for i in ids:
                case_dir = run_dirs[i]
                tree = openfoam_to_cgns_tree(case_dir)
                
                sample = Sample()
                sample.add_tree(tree)
                
                # Add global quantities from CSV
                # (matching logic as in sequential version)
                
                yield sample
    
    N_SHARD = N_PROC
    gen_kwargs = {
        "train": {'shards_ids': split_list(train_ids, N_SHARD)},
        "test": {'shards_ids': split_list(test_ids, N_SHARD)}
    }
    
    generators = {
        "train": _generator_parallel,
        "test": _generator_parallel
    }
    
    for backend in all_backends:
        print("--------------------------------------")
        print(f"Backend: {backend}, parallel version")
        
        repo_id = f"{BASE_REPO_ID}_{backend}"
        local_folder = f"{BASE_GENERATED_DATA_FOLDER}/drivaerml_{backend}"
        
        start = time.time()
        save_to_disk(
            output_folder=local_folder,
            generators=generators,
            backend=backend,
            infos=infos,
            pb_defs=pb_def,
            gen_kwargs=gen_kwargs,
            num_proc=N_PROC,
            overwrite=True,
            verbose=True
        )
        print(f"Duration: {time.time() - start:.2f} s")
        
        # Push to Hub
        push_to_hub(
            repo_id=repo_id,
            local_dir=local_folder,
            num_workers=N_PROC,
            viewer=(backend == "hf_datasets"),
            arxiv_paper_urls=["https://arxiv.org/abs/2408.11969"],
            pretty_name="DrivAerML - Road Car External Aerodynamics CFD Dataset"
        )


#---------------------------------------------------------------
# Notes for users
#---------------------------------------------------------------

"""
IMPLEMENTATION NOTES:

1. OpenFOAM Parsing:
   - The functions read_openfoam_polymesh() and read_openfoam_field() are
     placeholders and MUST be implemented by the user.
   - Consider using:
     * PyFoam: Python library for OpenFOAM case manipulation
     * Custom parsers for ASCII OpenFOAM format
     * Muscat with OpenFOAM support if available

2. Field Location:
   - OpenFOAM typically uses cell-centered fields (stored on cells, not nodes)
   - Use mesh.elemFields for cell-centered data
   - Use mesh.nodeFields for nodal data if applicable
   - Adjust feature identifiers accordingly (CellData vs VertexFields)

3. CSV Matching:
   - The geo_parameters_all.csv and force_mom_all.csv contain metadata
   - Implement proper matching between run directories and CSV rows
   - Add geometric parameters and force coefficients as global scalars

4. Mesh Complexity:
   - DrivAerML meshes are large and complex (automotive aerodynamics)
   - Consider processing a subset first for testing
   - Use parallel version for faster processing

5. Feature Engineering:
   - Users may want to add geometric parameters as input features
   - Consider adding boundary conditions or simulation parameters
   - Output features can include various force/moment coefficients

6. Dataset Structure:
   - Each Sample represents one static CFD simulation (steady-state)
   - No temporal dimension (see skills/patterns/static_vs_temporal_samples.md)
   - Multiple car geometries with different aerodynamic properties
"""
