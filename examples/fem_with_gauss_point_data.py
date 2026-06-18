"""Convert finite-element data with integration-point fields to PLAID.

This example reads a Zset ``.ut`` file, stores nodal and integration-point data in a :class:`plaid.Sample`, and verifies the result after writing and reading the data with the supported storage backends.
"""

import importlib.util
from pathlib import Path

import numpy as np
from Muscat.Bridges.CGNSBridge import (
    AddMuscatIPField,
    CGNSToMesh,
    ExtractIPField,
    MeshToCGNS,
)
from Muscat.FE.Fields.IPField import IPField
from Muscat.IO.UtReader import UtReader
from Muscat.IO.ZsetTools import GetIntegrationRuleForZsetMesh
from Muscat.MeshContainers import ElementsDescription as ED
from Muscat.MeshContainers.Filters.FilterObjects import ElementFilter
from Muscat.TestData import GetTestDataPath

from plaid import Sample
from plaid.storage import init_from_disk, save_to_disk


def isInstalled(module):
    """Return whether an importable module is available."""
    return importlib.util.find_spec(module) is not None


def isJupyter():
    """Return whether the current process is running in a Jupyter kernel."""
    get_ipython = globals().get("get_ipython")
    if get_ipython is None:
        return False
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except (NameError, AttributeError):
        return False


def main() -> None:
    """Convert, store, reload, and visualize integration-point data."""
    # Read finite-element data with integration-point information.
    ut_file_path = Path(GetTestDataPath()) / "UtExample" / "cube.ut"
    print(ut_file_path)

    reader = UtReader()
    reader.SetFileName(ut_file_path)
    reader.ReadMetaData()
    times = reader.GetAvailableTimes()
    print(times)

    # The mesh does not change, so load it only once.
    reader.SetTimeToRead(times[0])
    mesh = reader.Read()
    print(mesh)

    # CGNS does not support node and element tags with the same name.
    # Rename duplicate element tags by adding the "el_" prefix.
    mesh.nodeFields = {}
    for etag in mesh.elements.GetTagsNames():
        for el in mesh.elements:
            if etag in el.tags:
                el.tags.RenameTag(etag, "el_" + etag)
                print(f"rename tag {etag} -> {'el_' + etag} ")

    print(mesh)
    saved_mesh = mesh.View()

    # The mesh contains string element fields (the Zset finite-element names).
    # Remove the string fields before writing the mesh to CGNS.
    mesh.elemFields = {}

    # Display the data available in the .ut file.
    print("node data: ", reader.node)
    print("integration point data: ", reader.integ)
    print(reader.time[:, -1])

    # Create one independent CGNS tree for each time step.
    sample = Sample()
    # UtReader.time contains increments and steps; index 4 is the physical time.
    # Use a new CGNS tree for every step so the objects remain independent.
    # Use a mesh view so the NumPy arrays cannot be modified.
    for step_data in reader.time:
        t = step_data[4]
        sample.add_tree(MeshToCGNS(mesh.View()), time=t)

    print(sample)
    print(sample.get_field_names(time=0))

    # Add nodal and integration-point fields as vertex fields.
    reader.atIntegrationPoints = False
    for step_data in reader.time:
        t = step_data[4]
        print(t)
        reader.SetTimeToRead(t)
        sample.set_default_time(t)
        for node_field in reader.node:
            data = reader.ReadField(fieldname=node_field)
            sample.add_field(node_field, data, location="Vertex")

        for integ_field in reader.integ:
            data = reader.ReadField(fieldname=integ_field)
            sample.add_field(integ_field, data, location="Vertex")
    print(sample)

    # Add the integration-point fields at their native location.

    reader.atIntegrationPoints = True

    # keep track of fields for the check at the end of the file
    # this is only for verification
    allipfs = {}

    mesh_quadrature = GetIntegrationRuleForZsetMesh(saved_mesh)

    for step_data in reader.time:
        t = step_data[4]
        print(t)
        reader.SetTimeToRead(t)
        sample.set_default_time(t)

        ipfs = []
        for integ_field in reader.integ:
            data = reader.ReadField(fieldname=integ_field, time=t)
            bulk_filter = ElementFilter(dimensionality=mesh.GetElementsDimensionality())
            ipf = IPField(name=integ_field, mesh=mesh, rule=mesh_quadrature)
            ipf.Allocate()
            ipf.SetDataFromNumpy(data, bulk_filter)
            ipfs.append(ipf)

        # Also export the integration-point positions as fields.
        AddMuscatIPField(sample.get_tree(time=t), ipfs, exportLocationsPositions=True)

        allipfs[t] = {i.name: i for i in ipfs}

    #
    print(sample)

    #
    keys = list(sample.data.keys())
    values = list(sample.data.values())

    # Create a multi-sample database with one time step per sample.
    def temporal_sample_constructor(i: int):
        temp_sample = Sample()
        temp_sample.data = {keys[i]: values[i]}
        return temp_sample

    for backend in ["cgns", "hf_datasets", "zarr"]:
        print(
            "  ---------  " + backend + "  " + "-" * (11 - len(backend)) + "---------"
        )
        save_to_disk(
            output_folder=f"output_dataset_with_gauss_{backend}_per_step",
            sample_constructor=temporal_sample_constructor,
            ids={"train": list(range(len(keys)))},
            backend=backend,
            overwrite=True,
            num_proc=1,
        )

    # Save the single sample
    def sample_constructor(i: int):
        if i > 0:
            raise
        return sample

    # HF Datasets and Zarr do not support a one-sample dataset.
    for backend in ["cgns"]:
        print(
            "  ---------  " + backend + "  " + "-" * (11 - len(backend)) + "---------"
        )
        save_to_disk(
            output_folder=f"output_dataset_with_gauss_{backend}_one_sample",
            sample_constructor=sample_constructor,
            ids={"train": [0]},
            backend=backend,
            overwrite=True,
        )

    # Reload the data and verify that the integration-point fields are unchanged.

    for backend in ["cgns", "hf_datasets", "zarr"]:
        datasetdict, converterdict = init_from_disk(
            local_dir=f"output_dataset_with_gauss_{backend}_per_step"
        )

        for i, t in enumerate(keys):
            print(f"Working on backend {backend}, time {t}")
            sample_back = converterdict["train"].to_plaid(datasetdict["train"], i)
            print(1)
            mesh_back = CGNSToMesh(sample_back.get_tree(time=t))
            print(2)
            for f in sample_back.get_field_names(location="IntegrationPoint", time=t):
                # Skip the integration-point position fields.
                if any(f.endswith(x) for x in ["_posx", "_posy", "_posz"]):
                    continue
                print(f, end="")
                field = sample_back.get_field(f, "IntegrationPoint", time=t)
                reader.atIntegrationPoints = True
                # Read the original field to check consistency.
                data = reader.ReadField(fieldname=f, time=t)
                ipField_back = ExtractIPField(
                    sample_back.get_tree(time=t), mesh_back, f
                )
                check0 = np.allclose(field.shape, data.shape)
                check1 = np.allclose(field, data)
                check2 = np.allclose(
                    ipField_back.data[ED.Hexahedron_8],
                    allipfs[t][f].data[ED.Hexahedron_8],
                )

                if not (check0 and check1 and check2):
                    print(f, check0, check1, check2)
                    raise
                print(" OK")

    print("Original sample before export:", sample)
    print("Last sample after reloading:", sample_back)
    print("Done")

    # Recover the integration-point positions.
    eto11 = sample_back.get_field("eto11", "IntegrationPoint", time=keys[-1])
    eto11_posx = sample_back.get_field("eto11_posx", "IntegrationPoint", time=keys[-1])
    eto11_posy = sample_back.get_field("eto11_posy", "IntegrationPoint", time=keys[-1])
    eto11_posz = sample_back.get_field("eto11_posz", "IntegrationPoint", time=keys[-1])

    if isInstalled("pyvista"):
        import pyvista as pv

        point_cloud = pv.PolyData(np.vstack((eto11_posx, eto11_posy, eto11_posz)).T)
        point_cloud["eto11"] = eto11
        print(point_cloud)
        plotter = pv.Plotter()
        plotter.add_mesh(
            point_cloud,
            scalars="eto11",
            style="points",
            point_size=10.0,
        )
        plotter.show(jupyter_backend="static")
    else:
        print("PyVista is not installed; skipping the point-cloud plot.")

    if isInstalled("plotly"):
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=eto11_posx,
                    y=eto11_posy,
                    z=eto11_posz,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=eto11,
                        colorscale="Viridis",
                        colorbar=dict(title="Values"),
                    ),
                )
            ]
        )
        renderer = "jupyterlab" if isJupyter() else "browser"
        fig.show(renderer=renderer)

    # Plot the mesh.
    if isInstalled("pyvista"):
        from plaid.utils.cgns_vtk import CGNSTreeToVtk

        vtk_mesh = CGNSTreeToVtk(sample_back.get_tree(time=keys[-1]))
        if isJupyter():
            pv.set_jupyter_backend("static")
        plotter = pv.Plotter()
        plotter.add_mesh(vtk_mesh, scalars="U1", show_edges=True)
        plotter.show()


if __name__ == "__main__":
    main()
