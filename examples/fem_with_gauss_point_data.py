# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Example of converting user data into PLAID
#
# This code provides an example for converting user data into the PLAID (Physics Informed AI Datamodel) format.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

from Muscat.Bridges.CGNSBridge import MeshToCGNS, CGNSToMesh
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid import Sample

from Muscat.TestData import GetTestDataPath

ut_file_path = Path(GetTestDataPath())/ 'UtExample'/ 'cube.ut'
print(ut_file_path)

from Muscat.IO.UtReader import UtReader

reader = UtReader()
reader.SetFileName(ut_file_path)
reader.ReadMetaData()
times = reader.GetAvailableTimes()
print(times)
#%%
# we know the mesh does not change, load the mesh only ones
reader.SetTimeToRead(times[0])
mesh = reader.Read()
#print(mesh)
#exit()
#%% clean mesh
mesh.nodeFields = {}
for etag in mesh.elements.GetTagsNames():
    print(etag)
    for el in mesh.elements:
        if etag in el.tags:
            el.tags.RenameTag(etag, "el_"+etag)
print(mesh)

cgns_tree = MeshToCGNS(mesh)
#print(CGNSToMesh(cgns_tree))
#from Muscat.MeshTools.MeshTools import IsClose
#IsClose(mesh, CGNSToMesh(MeshToCGNS(mesh)))
#exit()
print(reader.node)
print(reader.integ)
print(reader.time)
sample = Sample()
for step_data in reader.time:
    print(step_data)
    t = step_data[4]
    mesh_i = mesh.View()
    reader.SetTimeToRead(t)
    reader.atIntegrationPoints = False
    cgns_tree = MeshToCGNS(mesh)


    sample.features.add_tree(cgns_tree, time=t)
    sample.set_default_time(t)

    for node_field in reader.node:
        data = reader.ReadField(fieldname= node_field)
        sample.add_field(node_field, data, location="Vertex")

    for integ_field in reader.integ:
        data = reader.ReadField(fieldname= integ_field)
        sample.add_field(integ_field, data, location="Vertex")

    reader.atIntegrationPoints = True

    from Muscat.IO.ZsetTools import GetIntegrationRuleForZsetMesh
    from Muscat.Bridges.CGNSBridge import MuscatToCGNSNames
    mesh_quadrature = GetIntegrationRuleForZsetMesh(mesh)
    idToName = {}
    ruleIdByName = {}
    rules = {}

    for i, (k,v) in enumerate(mesh_quadrature.items()):
        #print(i,k,v)
        name = f"zset{str(k).strip("ElementType.")}_IntRule"
        idToName[i] = name
        ruleIdByName[name] = i
        rules[name] = {
            "element_type": MuscatToCGNSNames[k],
            "reference_space": "Parametric",
            "integration_name": "Zset",
            "parametric_integration_points": np.asarray(v.points, order='F'),
                "weights": v.weights,
        }

    from Muscat.Bridges.CGNSBridge import AddIntegrationRuleCollection, AddIntegrationPointFlowSolution
    AddIntegrationRuleCollection(
        sample.features.get_base(),
        collectionName="IntegrationGaussZset",
        idToName=idToName,
        rules=rules,
    )

    for integ_field in reader.integ:
        data = reader.ReadField(fieldname= integ_field)

        from Muscat.MeshContainers.Filters.FilterObjects import ElementFilter
        bulk_filter = ElementFilter(dimensionality=mesh.GetElementsDimensionality())
        from Muscat.FE.Fields.IPField import IPField
        ipf = IPField(name=integ_field,mesh=mesh,rule = mesh_quadrature)
        ipf.Allocate()
        ipf.SetDataFromNumpy(data, bulk_filter)

        offset = ipf.GetFlattenOffset(bulk_filter)
        vals = ipf.Flatten(bulk_filter)

        nCells = ipf.mesh.GetNumberOfElements(dim=ipf.mesh.GetElementsDimensionality())
        itgIds = np.zeros(nCells, dtype=np.int32)
        for selection in bulk_filter(ipf.mesh):
            ruleName = f"zset{str(selection.elementType).strip("ElementType.")}_IntRule"
            itgIds[selection.GetSelectionSlice()] = ruleIdByName[ruleName]

        AddIntegrationPointFlowSolution(
                sample.features.get_zone(),
                flowName=f"{ipf.name}_IntegrationPointFields",
                dataArrays={ipf.name: vals},
                itgPointStartOffset=offset,
                itgRulesPath=f"/{sample.resolve_base()}/IntegrationGaussZset",
                itgRulesIds=itgIds,
            )

print(sample)

from plaid.storage import save_to_disk

keys = list(sample.features.data.keys())
values = list(sample.features.data.values())


sample.features.data = {}

def sample_constructor(i: int):
    sample.features.data = {}
    sample.features.data = {keys[i]:values[i]}
    return sample

for backend in ["cgns"]:#, "hf_datasets"]:#,"zarr"]:
    save_to_disk(
        output_folder=f"output_dataset_with_gauss_{backend}",
        sample_constructor=sample_constructor,
        ids={"train": [0]},# np.arange(len(keys))},
        backend=backend,      # or "hf_datasets" or "cgns"
        overwrite=True,
    )


from plaid.storage import init_from_disk


datasetdict, converterdict =  init_from_disk(
    local_dir = "output_dataset_with_gauss_cgns",
    splits =  ["train"]
)
sample0 = converterdict["train"].to_plaid(datasetdict["train"], 0)
sample.features.data = {keys[0]:values[0]}

print(mesh)
mesh2 = CGNSToMesh(sample0.features.data[0.0])
print(mesh2)
print(set(mesh.nodesTags.keys()) - set(mesh2.nodesTags.keys()))

mesh3 = CGNSToMesh(MeshToCGNS(mesh))
print(mesh3)
print(set(mesh.nodesTags.keys()) - set(mesh3.nodesTags.keys()))
exit()

with open("sample0", "w") as f:
    f.write('"""from the backend"""')
    with open("sample", "w") as f2:
        f2.write('"""from User"""')
        def pprint(a,b,offset=0):
            if isinstance(a,list):
                a_names = ((n[0],n[3]) for n in a)
                b_names = ((n[0],n[3]) for n in b)


                for n in (set(a_names) | set(b_names)):
                    a_result = list(filter(lambda u: (u[0],u[3])== n, a))
                    b_result = list(filter(lambda u: (u[0],u[3])== n, b))


                    if len(a_result) < 1 :
                        print (f"error {n} in a ")
                        f.write(" "*offset+str(n)+ " not found in a \n")
                        f2.write(" "*offset+str(b_result)+ " found in b \n")
                        continue
                    if len(b_result) < 1:
                        print(b_result)
                        f.write(" "*offset+str(a_result[0])+ " found in a \n")
                        f2.write(" "*offset+str(n)+ " not found in b \n")

                        print( f"error {n} in b ")
                        continue
                    f.write(" "*offset+str(a_result[0][0])+ "\n")
                    f.write(" "*offset+str(a_result[0][1])+ "\n")
                    f.write(" "*offset+str(a_result[0][3])+ "\n")

                    f2.write(" "*offset+str(b_result[0][0])+ "\n")
                    f2.write(" "*offset+str(b_result[0][1])+ "\n")
                    f2.write(" "*offset+str(b_result[0][3])+ "\n")


                    pprint(a_result[0][2],b_result[0][2], offset+2)
            else:
                f.write(" "*offset+str(a)+ "\n")
                f2.write(" "*offset+str(b)+ "\n")
            #for i in a :
            #    f.write(str(i))
            #for i in b :
            #    f2.write(str(i))
        pprint([sample0.features.data[0.0]],[sample.features.data[0.0]])
print("cone")
exit()


for f in sample.get_all_features_identifiers_by_type("field"):

    field  = sample.get_field(f, "IntegrationPoint")
    if field is not None:
        print(f, field.shape)

version = 1
import pickle
with open("fromUt.pickle", "wb") as f:
    if version == 0:
        pickle.dump(0,f)
        pickle.dump(sample.features.get_all_time_values(),f)
        pickle.dump(sample,f)

    if version == 1:
        timesteps = sample.features.get_all_time_values()
        sizes = np.empty(len(timesteps)+1,dtype=int)
        pickle.dump(1,f)
        init = f.tell()
        pickle.dump((timesteps,sizes),f)
        for i in range(len(timesteps)):
            sizes[i] = f.tell()
            data = sample.features.data[timesteps[i]]
            pickle.dump(data,f)
        sample.features.data = None
        sizes[-1] = f.tell()
        pickle.dump(sample,f)
        f.seek(init)
        pickle.dump((timesteps,sizes),f)

if version == 0:
    with open("fromUt.pickle", "rb") as f:
        # drop version
        pickle.load(f)
        # drop timevalues
        pickle.load(f)
        res  = pickle.load(f)

if version == 1:
    with open("fromUt.pickle", "rb") as f:
        print("version", pickle.load(f))
        timestaps , offsets = pickle.load(f)
        data = {}
        for i,(t,off) in enumerate(zip(timestaps,offsets)):
            f.seek(off)
            data[t] = pickle.load(f)
        f.seek(offsets[-1])
        res = pickle.load(f)
        res.features.data = data


