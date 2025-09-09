
from plaid import Sample
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
import copy
from tqdm import tqdm

import os, shutil
import time


plaid_location = # path to update
prepared_data_dir= # path to update


start = time.time()


from plaid import ProblemDefinition


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



in_scalars_names = ["C11","C12","C22"]
out_fields_names = ["u1", "u2", "P11", "P12", "P22", "P21", "psi"]
out_scalars_names = ["effective_energy"]


problem = ProblemDefinition()
problem._load_from_dir_(pb_defpath)

ids_train = problem.get_split('DOE_train')
ids_test  = problem.get_split('DOE_test')



size=200
rec_mesh = Tetrahedrization(CreateConstantRectilinearMesh([size+1,size+1], [0,0], [1/size, 1/size]))
out_nodes = rec_mesh.nodes


nSamples = len(ids_train)+len(ids_test)

for sample_index in tqdm(range(nSamples)):

    sample = Sample.load_from_dir(dir_path = os.path.join(datapath, "samples/sample_{:09d}".format(sample_index)))

    input_mesh = CGNSToMesh(sample.get_mesh(time=0))

    input_mesh.nodes= (input_mesh.nodes[:,[0,1]]).copy(order='C')

    new_sample=Sample()
    tree = MeshToCGNS(rec_mesh)
    new_sample.add_mesh(tree,time=0)


    if sample_index in ids_train:
        scalar_names = in_scalars_names + out_scalars_names

        space, numberings,_,_ = PrepareFEComputation(input_mesh,numberOfComponents=1)
        field_mesh = FEField("", mesh=input_mesh, space=space, numbering=numberings[0])
        efilter = ElementFilter(dimensionality=input_mesh.GetElementsDimensionality())
        op, status, _  =  GetFieldTransferOp(inputField= field_mesh, targetPoints= out_nodes, method="Interp/Clamp" , elementFilter=efilter  , verbose=False)

        for field_name in out_fields_names:
            old_field = sample.get_field( name=field_name)
            new_sample.add_field(field_name, op.dot(old_field))

    elif sample_index in ids_test:
        scalar_names = in_scalars_names
    else:
        raise("unkown sample_index")


    for scalar_name in scalar_names:
        old_scalar= sample.get_scalar( name=scalar_name)
        new_sample.add_scalar(scalar_name, old_scalar)
    new_sample.add_field("Signed_Distance",compute_signed_distance(copy.deepcopy(input_mesh),rec_mesh.nodes))

    path = os.path.join(prepared_data_dir,"dataset/samples/sample_{:09d}".format(sample_index))
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    new_sample.save(path)

print("duration prepare =", time.time()-start)
# 126 seconds
