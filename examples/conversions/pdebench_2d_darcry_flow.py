"""
PLAID Dataset Conversion Example

Dataset:
- PDEBench: 2D Darcy Flow (beta-parameterized variants)

Purpose:
- Convert a static, parameterized PDE dataset with cell-centered fields
  on a structured rectilinear grid to the PLAID format.

Notes:
- Each Sample represents one independent static simulation
- No temporal dimension is present
- Fields are cell-centered (element-based), not nodal
- A single constant mesh is reused across all samples
- Multiple datasets are generated via a parameter sweep at script level
- External dependencies are required (e.g. h5py, Muscat)
- Script is not meant to be generic or reusable as-is
"""

from pathlib import Path

import h5py
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools.ConstantRectilinearMeshTools import CreateConstantRectilinearMesh
from plaid import ProblemDefinition, Sample
from plaid.storage import push_to_hub, save_to_disk

root = Path(__file__).parent.parent.resolve()
BACKENDS = ["hf_datasets"]


download_dir = "/path/downloaded/file"

assert download_dir != "/path/downloaded/file", "Please set download_dir"


def main(beta: float = 0.01):
    DATA_FILE = (
        f"{download_dir}/2D_DarcyFlow_beta{beta}_Train.hdf5"
    )

    mesh = CreateConstantRectilinearMesh(
        dimensions=[129, 129],
        origin=[0.0, 0.0],
        spacing=[1.0 / 128.0, 1.0 / 128.0],
    )

    infos = {
        "legal": {
            "owner": "Takamoto, M et al. (https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)",
            "license": "cc-by-4.0",
        },
        "data_production": {
            "physics": "2D Darcy Flow",
            "type": "simulation",
            "script": "Converted to PLAID format for standardized usage; no changes to data content.",
        },
    }

    constant_features = [
        "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
        "Base_2_2/Zone/Elements_QUAD_4/ElementRange",
        "Base_2_2/Zone/GridCoordinates/CoordinateX",
        "Base_2_2/Zone/GridCoordinates/CoordinateY",
        "Base_2_2/Zone/PointData/GridLocation",
        "Global/forcing_magnitude",
    ]
    input_features = [
        "Base_2_2/Zone/CellData/diffusion_coefficient",
    ]
    output_features = [
        "Base_2_2/Zone/CellData/flow",
    ]

    pb_def = ProblemDefinition()
    pb_def.add_in_features_identifiers(input_features)
    pb_def.add_out_features_identifiers(output_features)
    pb_def.add_constant_features_identifiers(constant_features)
    pb_def.set_task("regression")
    pb_def.set_split({"all_samples": list(range(10_000))})

    with h5py.File(DATA_FILE, "r") as f:
        perm = f["nu"][:]
        output = f["tensor"][:]

        def generator_fn():
            for i in range(perm.shape[0]):
                mesh.elemFields = {}
                mesh.elemFields["diffusion_coefficient"] = perm[i, :, :].reshape(-1)
                mesh.elemFields["flow"] = output[i, 0, :, :].reshape(-1)

                tree = MeshToCGNS(mesh)

                sample = Sample()
                sample.features.add_tree(tree)
                sample.add_scalar(name="forcing_magnitude", value=0.1)

                yield sample

        generators = {"train": generator_fn}

        for backend in BACKENDS:
            print(f"Backend: {backend}, sequential version")

            local_folder = f"plaid/PDEBench_2D_DarcyFlow_beta{beta}_{backend}"

            save_to_disk(
                output_folder=local_folder,
                generators=generators,
                backend=backend,
                infos=infos,
                pb_defs=pb_def,
                overwrite=True,
                verbose=True,
            )

            if backend == "hf_datasets":
                push_to_hub(
                    repo_id=f"Nionio/PDEBench_2D_DarcyFlow_beta{beta}",
                    local_dir=local_folder,
                    viewer=True,
                    arxiv_paper_urls=["https://arxiv.org/pdf/2210.07182"],
                    pretty_name=f"PDEBench 2D Darcy Flow with forcing {beta}",
                )


if __name__ == "__main__":
    for beta_value in [0.01, 0.1, 1.0, 10.0, 100.0]:
        print(f"Processing beta={beta_value}")
        main(beta=beta_value)
    main()