from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from Muscat.Bridges.CGNSBridge import CGNSToMesh
import numpy as np
from Muscat.Containers.Filters.FilterObjects import ElementFilter
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.Fields.FEField import FEField
from Muscat.Bridges.CGNSBridge import MeshToCGNS,CGNSToMesh
from Muscat.Containers.ConstantRectilinearMeshTools import CreateConstantRectilinearMesh
from Muscat.Containers.MeshTetrahedrization import Tetrahedrization
from Muscat.Containers.MeshModificationTools import ComputeSkin
from Muscat.FE.FETools import PrepareFEComputation
from Muscat.FE.FETools import ComputeNormalsAtPoints

import os, shutil, time
from tqdm import tqdm




plaid_location = # path to update
prepared_data_dir = # path to update




start = time.time()


def compute_signed_distance(mesh,eval_points):
    """Function to compute the signed distance from the border of the mesh

    Args:
        mesh (Muscat.Mesh): mesh which needs the singed distance
        eval_points (np.array): Points where to compute the signed distance.

    Returns:
        np.array: returns the signed distance of the mesh at th eeval points
    """
    ComputeSkin(mesh,inPlace=True)
    space, numberings,_,_ = PrepareFEComputation(mesh,numberOfComponents=1)
    field_mesh = FEField("", mesh=mesh, space=space, numbering=numberings[0])
    opSkin, statusSkin, _  =  GetFieldTransferOp(inputField= field_mesh, targetPoints= eval_points, method="Interp/Clamp" , elementFilter= ElementFilter(dimensionality=mesh.GetElementsDimensionality()-1) , verbose=False)
    normals = ComputeNormalsAtPoints(mesh)
    skinpos = opSkin.dot(mesh.nodes)
    normalspos = opSkin.dot(normals)
    sign_distance = -1*np.sign(np.sum((eval_points - skinpos)*normalspos,axis=-1))
    distance = np.sqrt(np.sum((eval_points - skinpos)**2,axis=-1))
    return sign_distance*distance



datapath=os.path.join(plaid_location, "dataset")
pb_defpath=os.path.join(plaid_location, "problem_definition")


problem = ProblemDefinition()
problem._load_from_dir_(pb_defpath)

ids_train = problem.get_split('train_1000')
ids_test  = problem.get_split('test')



in_scalars_names = ['Omega', 'P']
out_fields_names = ['Density', 'Pressure', 'Temperature']
out_scalars_names = ['Massflow', 'Compression_ratio', 'Efficiency']


size = 64

length = [0.06/(size-1), 0.083/(size-1), 0.06/(size-1)]
origin = [-0.01, 0.172, -0.03]

ref_mesh = CreateConstantRectilinearMesh([size,size,size], origin, length)


out_nodes = ref_mesh.nodes

for sample_index in tqdm(range(len(ids_train)+len(ids_test))):

    sample = Sample.load_from_dir(dir_path = os.path.join(datapath, "samples/sample_{:09d}".format(sample_index)))
    input_mesh = CGNSToMesh(sample.get_mesh(time=0))


    new_sample=Sample()
    tree = MeshToCGNS(ref_mesh)
    new_sample.add_tree(tree)


    if sample_index in ids_train:
        scalar_names = in_scalars_names + out_scalars_names

        space, numberings,_,_ = PrepareFEComputation(input_mesh,numberOfComponents=1)
        field_mesh = FEField("", mesh=input_mesh, space=space, numbering=numberings[0])
        efilter = ElementFilter(dimensionality=input_mesh.GetElementsDimensionality())
        op, status, _  =  GetFieldTransferOp(inputField= field_mesh, targetPoints= out_nodes, method="Interp/Clamp", elementFilter=efilter, verbose=False)

        for fn in out_fields_names:
            projected_field = op.dot(sample.get_field(fn))
            new_sample.add_field(fn, projected_field)

    elif sample_index in ids_test:
        scalar_names = in_scalars_names
    else:
        raise("unkown sample_index")


    skinpos = op.dot(input_mesh.nodes)
    sign_distance = np.linalg.norm(out_nodes - skinpos,axis=-1)

    new_sample.add_field("Signed_Distance", sign_distance)


    for scalar_name in scalar_names:
        old_scalar= sample.get_scalar( name=scalar_name)
        new_sample.add_scalar(scalar_name, old_scalar)


    path = os.path.join(prepared_data_dir,"dataset/samples/sample_{:09d}".format(sample_index))
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    new_sample.save(path)


print("duration prepare =", time.time()-start)
# 1689 seconds
