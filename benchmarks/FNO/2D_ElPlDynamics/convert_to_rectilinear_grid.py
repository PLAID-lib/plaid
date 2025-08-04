from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from Muscat.Bridges.CGNSBridge import CGNSToMesh
import numpy as np
from Muscat.MeshTools.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.Fields.FEField import FEField
from Muscat.FE.Spaces.FESpaces import LagrangeSpaceGeo
from Muscat.FE.DofNumbering import ComputeDofNumbering
from Muscat.Bridges.CGNSBridge import MeshToCGNS, CGNSToMesh
import Muscat.MeshContainers.ElementsDescription as ED
from Muscat.MeshTools.ConstantRectilinearMeshTools import CreateConstantRectilinearMesh
from Muscat.MeshTools.MeshTetrahedrization import Tetrahedrization
from Muscat.MeshTools.MeshModificationTools import DeleteElements
from Muscat.MeshTools.MeshTools import ComputeSignedDistance
from scipy.sparse import coo_matrix
import copy
import os


datapath = "/path/to/plaid/dataset"
output_data_dir = "/path/where/to/register/new_dataset"
dataset = Dataset()
for sample_index in range(1000):
    dataset._load_from_dir_(
        savedir=datapath,
        verbose=True,
        processes_number=1,
        ids=[sample_index])
    sample = dataset.get_samples([sample_index])[sample_index]
    old_mesh = CGNSToMesh(sample.get_mesh(time=0))

    old_mesh.nodes = (old_mesh.nodes[:, [0, 1]]).copy(order='C')
    size = 150
    ref_mesh = Tetrahedrization(CreateConstantRectilinearMesh(
        [size * 2 + 1, size + 1], [0, 0], [100 / size, 100 / size]))

    indexes = np.zeros(ref_mesh.GetNumberOfNodes(), int)
    values = np.zeros((ref_mesh.GetNumberOfNodes(), 15))
    for connect in ref_mesh.GetElementsOfType(ED.Triangle_3).connectivity:
        values[connect, indexes[connect]] = 1
        indexes[connect] += 1
    average = 1 / indexes
    data, i, j = np.zeros(0), np.zeros(
        0).astype(int), np.zeros(0).astype(int)
    for elem_index, connect in enumerate(
            ref_mesh.GetElementsOfType(ED.Triangle_3).connectivity):
        data = np.concatenate((data, average[connect]), axis=0)
        i = np.concatenate((i, connect), axis=0)
        j = np.concatenate((j, [elem_index] * 3), axis=0)

    operator_elem_to_node = coo_matrix((data, (i, j)))

    # Compute Field Transfer operator for node fields
    space = LagrangeSpaceGeo
    numbering = ComputeDofNumbering(old_mesh, space, fromConnectivity=True)
    displacement_field = FEField("FakeField", old_mesh, space, numbering)
    op, _, _ = GetFieldTransferOp(
        displacement_field, ref_mesh.nodes, method="Interp/Clamp")

    # Compute elem to node operator
    triangle_centers = np.mean(
        ref_mesh.nodes[ref_mesh.GetElementsOfType(ED.Triangle_3).connectivity], axis=-2)
    # entities contains the index of the element for each triangle_center
    _, _, entities = GetFieldTransferOp(
        displacement_field, triangle_centers, method="Interp/Clamp")
    data = np.ones_like(entities.squeeze())
    i = old_mesh.GetElementsOfType(ED.Triangle_3).GetNumberOfElements()
    op_elem = coo_matrix(
        (data, (np.arange(entities.shape[0]), entities.squeeze())))
    # Building a new sample
    new_sample = Sample()
    tree = MeshToCGNS(ref_mesh)
    new_sample.add_tree(tree, time=0)

    ux = sample.get_field(name="U_x", zone_name="Zone",
                          base_name="Base_2_3", time=0)
    uy = sample.get_field(name="U_y", zone_name="Zone",
                          base_name="Base_2_3", time=0)
    new_sample.add_field("U_x", op @ ux, zone_name="Zone",
                         base_name="Base_2_2", time=0)
    new_sample.add_field("U_y", op @ uy, zone_name="Zone",
                         base_name="Base_2_2", time=0)
    new_sample.add_field(
        "Signed_Distance",
        ComputeSignedDistance(
            copy.deepcopy(old_mesh),
            ref_mesh.nodes),
        zone_name="Zone",
        base_name="Base_2_2",
        time=0)
    dt = 0.001
    for i in range(1, 40):
        # Casting CGNS to Muscat.mesh
        old_mesh = CGNSToMesh(
            sample.get_mesh(
                time=dt * i,
                apply_links=True))
        old_mesh.nodes = (old_mesh.nodes[:, [0, 1]]).copy(order='C')

        # Compute Field Transfer operator for node fields
        space = LagrangeSpaceGeo
        numbering = ComputeDofNumbering(
            old_mesh, space, fromConnectivity=True)
        displacement_field = FEField(
            "FakeField", old_mesh, space, numbering)
        op, _, _ = GetFieldTransferOp(
            displacement_field, ref_mesh.nodes, method="Interp/ZeroFill")

        # Compute Field Transfer operator for elem fields
        triangle_centers = np.mean(
            ref_mesh.nodes[ref_mesh.GetElementsOfType(ED.Triangle_3).connectivity], axis=-2)
        _, _, entities = GetFieldTransferOp(
            displacement_field, triangle_centers, method="Interp/Clamp")
        data = np.ones_like(entities.squeeze())
        list_i = old_mesh.GetElementsOfType(
            ED.Triangle_3).GetNumberOfElements()
        op_elem = coo_matrix(
            (data, (np.arange(entities.shape[0]), entities.squeeze())))

        # Removing broken elements of the mesh
        mask = np.zeros(old_mesh.GetNumberOfElements())
        mask[old_mesh.elemFields['EROSION_STATUS'] == 0] = 1
        DeleteElements(old_mesh, mask, updateElementFields=True)

        ux = old_mesh.nodeFields["U_x"]
        uy = old_mesh.nodeFields["U_y"]
        path_linked_sample = os.path.join(
            output_data_dir, f"dataset/samples/sample_{sample_index:09d}/meshes/mesh_{0:09d}.cgns")
        new_sample.link_tree(
            path_linked_sample,
            linked_sample=new_sample,
            linked_time=0,
            time=dt * i)
        new_sample.add_field("U_x", op @ ux, zone_name="Zone",
                             base_name="Base_2_2", time=dt * i)
        new_sample.add_field("U_y", op @ uy, zone_name="Zone",
                             base_name="Base_2_2", time=dt * i)
        # Compute signed distance of the mesh where broken elements were
        # removed
        new_sample.add_field(
            "Signed_Distance",
            ComputeSignedDistance(
                copy.deepcopy(old_mesh),
                ref_mesh.nodes),
            zone_name="Zone",
            base_name="Base_2_2",
            time=dt * i)
        old_mesh = CGNSToMesh(
            sample.get_mesh(
                time=dt * i,
                apply_links=True))
    new_sample.save(os.path.join(output_data_dir,
                    "dataset/samples/sample_{:09d}".format(sample_index)))
