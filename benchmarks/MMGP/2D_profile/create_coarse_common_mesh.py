from Muscat.IO import XdmfWriter as XW
from Muscat.MeshTools.Remesh import Remesh
from Muscat.Containers import AnisotropicMetricComputation as AMC
import numpy as np
from Muscat.Containers.MeshModificationTools import ComputeSkin


from Muscat.Containers import ElementsDescription as ED
from Muscat.Containers.Filters import FilterObjects as FO
from Muscat.Containers.Filters import FilterOperators as FOp

from Muscat.IO.CGNSReader import ReadCGNS

import os


plaid_location = # path to update to input plaid dataset


reference_mesh_index = 0

common_mesh_path = os.path.join(plaid_location, "dataset/samples/sample_00000000"+str(reference_mesh_index)+"/meshes/mesh_000000000.cgns")
mesh = ReadCGNS(fileName=common_mesh_path)


mesh.nodes = np.ascontiguousarray(mesh.nodes[:,:2])

m = mesh.nodeFields['Mach']
p = mesh.nodeFields['Pressure']
Ux = mesh.nodeFields['Velocity-x']
Uy = mesh.nodeFields['Velocity-y']


m_scaled = (1.0 / np.std(m)) * (m - np.mean(m))
p_scaled = (1.0 / np.std(p)) * (p - np.mean(p))
norm_U = np.sqrt(Ux**2 + Uy**2)
norm_U_scaled = (1.0 / np.std(norm_U)) * (norm_U - np.mean(norm_U))
psi = 1.*p_scaled + 1.*m_scaled + 0.*norm_U_scaled

metric = AMC.ComputeMetric(mesh, psi, err = 0.03, gradL2 = False)
# metric = AMC.ComputeMetric(mesh, psi, err = 0.03, hmin = 0.005, hmax = 1000, gradL2 = False)

remeshed_mesh = Remesh(mesh = mesh, solution = None, metric = metric)

remeshed_mesh = ComputeSkin(remeshed_mesh, inPlace=True)

nf_skin = FO.NodeFilter(eTag =["Skin"])
indices_skin = nf_skin.GetNodesIndices(remeshed_mesh)

nf1 = FO.NodeFilter(eTag =["Skin"], zone=[lambda xyz: (xyz[:, 0] - 1.5)])
nf2 = FO.NodeFilter(eTag =["Skin"], zone=[lambda xyz: (xyz[:, 1] - 0.5)])
nf3 = FO.NodeFilter(eTag =["Skin"], zone=[lambda xyz: (-xyz[:, 0] - 0.5)])
nf4 = FO.NodeFilter(eTag =["Skin"], zone=[lambda xyz: (-xyz[:, 1] - 0.5)])


nf = FOp.IntersectionFilter(filters=[nf1, nf2, nf3, nf4])
indices_airfoil = nf.GetNodesIndices(remeshed_mesh)
# print("indices_airfoil =", indices_airfoil)
remeshed_mesh.GetNodalTag("Airfoil").AddToTag(indices_airfoil)

indices_ext_bound = np.setdiff1d(indices_skin, indices_airfoil)
remeshed_mesh.GetNodalTag("Ext_bound").AddToTag(indices_ext_bound)

nf5 = FO.NodeFilter(eTag =["Skin"], zone=[lambda xyz: (xyz[:, 0] + 0.9999)])
inlet_indices = nf5.GetNodesIndices(remeshed_mesh)
remeshed_mesh.GetNodalTag("Inlet").AddToTag(inlet_indices)

remeshed_mesh.elements[ED.Bar_2].tags.DeleteTags(["Skin"])
del remeshed_mesh.elements[ED.Bar_2]
remeshed_mesh.nodesTags.DeleteTags(['Corners', 'NTag_mmg_0'])


XW.WriteMeshToXdmf("coarse_common_mesh.xdmf", remeshed_mesh)
