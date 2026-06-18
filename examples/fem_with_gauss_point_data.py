# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example of converting user data into PLAID with integration point data
#
# This code provides an example for converting user data into the PLAID (Physics Informed AI Datamodel) format.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path
import numpy as np

from Muscat.Bridges.CGNSBridge import MeshToCGNS, CGNSToMesh
from Muscat.TestData import GetTestDataPath
from Muscat.IO.UtReader import UtReader
from Muscat.Bridges.CGNSBridge import AddIntegrationPointFlowSolution, AddMuscatIPField
from Muscat.IO.ZsetTools import GetIntegrationRuleForZsetMesh
from Muscat.MeshContainers.Filters.FilterObjects import ElementFilter
from Muscat.FE.Fields.IPField import IPField
from Muscat.MeshContainers import ElementsDescription as ED
from Muscat.Bridges.CGNSBridge import ExtractIPField

from plaid import Sample
from plaid.storage import save_to_disk
from plaid.storage import init_from_disk


# %%
def main() -> None:


    # %%
    ut_file_path = Path(GetTestDataPath())/ 'UtExample'/ 'cube.ut'
    print(ut_file_path)

    reader = UtReader()
    reader.SetFileName(ut_file_path)
    reader.ReadMetaData()
    times = reader.GetAvailableTimes()
    print(times)
# %% [markdown]
#     # ## We know the mesh does not change, load the mesh only ones

    # %%
    reader.SetTimeToRead(times[0])
    mesh = reader.Read()
    print(mesh)

# %% [markdown]
#     # ## CGNS does not support nodes Tags and Elements tags with the same name

    # %%
    mesh.nodeFields = {}
    for etag in mesh.elements.GetTagsNames():

        for el in mesh.elements:
            if etag in el.tags:
                el.tags.RenameTag(etag, "el_"+etag)
                print(f"rename tag {etag} -> {'el_'+etag} ")

    print(mesh)
    saved_mesh = mesh.View()
    # removing string field
    mesh.elemFields = {}


# %% [markdown]
#     # ## Convert mesh to cgns tree

    # %%
    #print(CGNSToMesh(MeshToCGNS(mesh)))

# %% [markdown]
#     # ## Data  in the .ut file

    # %%
    print(reader.node)
    print(reader.integ)
    print(reader.time[:,-1])

# %% [markdown]
#     # ## Loop over the time steps and inject the mesh

    # %%
    sample = Sample()
    for step_data in reader.time:
        t = step_data[4]
        print(t)
        reader.SetTimeToRead(t)
        reader.atIntegrationPoints = False
        cgns_tree = MeshToCGNS(mesh)


        sample.add_tree(cgns_tree, time=t)
        sample.set_default_time(t)
    print(sample)


    # %%
    sample.get_field_names(time=0)

# %% [markdown]
#     # ## Loop over the time steps and inject vertex fields

    # %%
    reader.atIntegrationPoints = False
    for step_data in reader.time:
        t = step_data[4]
        print(t)
        reader.SetTimeToRead(t)

        sample.set_default_time(t)
        for node_field in reader.node:
            data = reader.ReadField(fieldname= node_field)
            sample.add_field(node_field, data, location="Vertex")

        for integ_field in reader.integ:
            data = reader.ReadField(fieldname= integ_field)
            sample.add_field(integ_field, data, location="Vertex")
    print(sample)

# %% [markdown]
#     # ## Loop over the fields and inject integration point data

    # %%
    reader.atIntegrationPoints = True

    # keep track of fields for the check at the end of the file
    ipdata = {}
    allipfs = {}
    mesh_quadrature = GetIntegrationRuleForZsetMesh(saved_mesh)

    for step_data in reader.time:
        t = step_data[4]
        print(t)
        reader.SetTimeToRead(t)
        sample.set_default_time(t)

        ipfs = []
        ipdata2  = {}
        ipdata[t] = ipdata2
        for integ_field in reader.integ:
            data = reader.ReadField(fieldname= integ_field, time = t)
            bulk_filter = ElementFilter(dimensionality=mesh.GetElementsDimensionality())
            ipf = IPField(name=integ_field,mesh=mesh,rule = mesh_quadrature)
            ipf.Allocate()
            ipf.SetDataFromNumpy(data, bulk_filter)
            ipdata2[integ_field] = data
            ipfs.append(ipf)

        ## User can export the integration point positions as ip fields
        AddMuscatIPField(sample.get_tree(time=t), ipfs, exportLocationsPositions=True)

        allipfs[t] = {i.name:i for i in ipfs}


    # %%
    print(sample)
    #sample.show_tree()

# %% [markdown]
#     # ## Store one step per sample

    # %%
    keys = list(sample.data.keys())
    values = list(sample.data.values())

    def sample_constructor(i: int):
        temp_sample = Sample()
        temp_sample.data = {keys[i]:values[i]}
        return temp_sample

    for backend in ["cgns","hf_datasets"]:#,"zarr" for the moment zarr froze for unkwnon reason during reading
        print(backend + "--------------------------------------------")

        save_to_disk(
            output_folder=f"output_dataset_with_gauss_{backend}_per_step",
            sample_constructor=sample_constructor,
            ids={"train": list(range(len(keys)))},
            backend=backend,
            overwrite=True,
            num_proc=1
        )


# %% [markdown]
#     # ## Store one time varing sample

# %%

    def sample_constructor(i: int):
        if i > 0:
            raise
        return sample

    for backend in ["cgns"]:# ,"hf_datasets","zarr"do not support 1 sample dataset
        print(backend + "--------------------------------------------")
        save_to_disk(
            output_folder=f"output_dataset_with_gauss_{backend}_one_sample",
            sample_constructor=sample_constructor,
            ids={"train": [0]},
            backend=backend,
            overwrite=True,
        )


# %% [markdown]
#     # ## Reload Data from disk and verify the integration point data is the same

# %% clean mesh

    for backend in ["cgns","hf_datasets"]:#,"zarr"
        datasetdict, converterdict =  init_from_disk(local_dir = f"output_dataset_with_gauss_{backend}_per_step")

        for i, t in enumerate(keys):
            print(f"Working on backend {backend}, time {t}")
            sample_back = converterdict["train"].to_plaid(datasetdict["train"], i)
            print(1)
            mesh_back = CGNSToMesh(sample_back.get_tree(time=t))
            print(2)
            for f in sample_back.get_field_names(location="IntegrationPoint", time=t):
                print(f)
                field  = sample_back.get_field(f, "IntegrationPoint", time=t)
                if f not in ipdata[t]:
                    check0 = np.allclose(field.shape, ipdata[t][f[0:-5]].shape)
                    check1 = "Na"
                    check2 = "Na"
                else:
                    ipField_back = ExtractIPField(sample_back.get_tree(time=t), mesh_back, f)
                    check0 = np.allclose(field.shape, ipdata[t][f].shape)
                    check1 = np.allclose(field,ipdata[t][f])
                    check2 = np.allclose(ipField_back.data[ED.Hexahedron_8],allipfs[t][f].data[ED.Hexahedron_8] )

                if not (check0 and check1 and check2) :
                    print(f, check0, check1, check2              )
                    raise

        print(sample)
    print(sample_back)
    print("Done")


# %% [markdown]
#     # ## Recover position of the integration Points

    # %%
    eto11 =sample_back.get_field('eto11',"IntegrationPoint",time=keys[-1])
    eto11_posx =sample_back.get_field('eto11_posx',"IntegrationPoint",time=keys[-1])
    eto11_posy =sample_back.get_field('eto11_posy',"IntegrationPoint",time=keys[-1])
    eto11_posz =sample_back.get_field('eto11_posz',"IntegrationPoint",time=keys[-1])


    # %%
    import pyvista as pv

    point_cloud = pv.PolyData(np.vstack((eto11_posx,eto11_posy,eto11_posz)).T)

    point_cloud["eto11"] = eto11
    print(point_cloud)
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="eto11", style="points", point_size=10.0)
    plotter.show(jupyter_backend='static')

    # %%
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
        x=eto11_posx, y=eto11_posy, z=eto11_posz,
        mode='markers',
        marker=dict(
            size=8,
            color=eto11,                # Pass numeric array here
            colorscale='Viridis',           # Choose color palette (capitalized)
            colorbar=dict(title="Values"),  # Shows the color legend side-bar
        )
    )])

    renderer = "notebook"
    fig.show(renderer=renderer)

    # %%
    from plaid.utils.cgns_vtk import CGNSTreeToVtk
    vtkmesh = CGNSTreeToVtk(sample_back.get_tree(time= keys[-1]))
    #print(vtkmesh)

    # %%
    pv.set_jupyter_backend('static')
    pl = pv.Plotter()
    pl.add_mesh(vtkmesh, scalars="U1", show_edges=True)
    pl.show()

# %%

# %%

if __name__ == "__main__":
    main()
