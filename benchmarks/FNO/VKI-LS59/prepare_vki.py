from plaid.problem_definition import ProblemDefinition
from plaid.containers.sample import Sample
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers.ConstantRectilinearMeshTools import CreateConstantRectilinearMesh
from Muscat.Containers.MeshTetrahedrization import Tetrahedrization
import os, time, shutil
from tqdm import tqdm


start = time.time()

plaid_location = # path to update
prepared_data_dir = # path to update




datapath=os.path.join(plaid_location, "dataset")
pb_defpath=os.path.join(plaid_location, "problem_definition")


problem = ProblemDefinition()
problem._load_from_dir_(pb_defpath)

ids_train = problem.get_split('train')
ids_test  = problem.get_split('test')


in_scalars_names = ['angle_in', 'mach_out']
out_fields_names = ['mach', 'nut']
out_scalars_names = ['Q', 'power', 'Pr', 'Tr', 'eth_is', 'angle_out']


nx = 301
ny = 121



rec_mesh = Tetrahedrization(CreateConstantRectilinearMesh([nx,ny], [0,0], [1/(nx-1), 1/(ny-1)]))
out_nodes = rec_mesh.nodes


nSamples = len(ids_train)+len(ids_test)

for sample_index in tqdm(range(nSamples)):

    sample = Sample.load_from_dir(dir_path = os.path.join(datapath, "samples/sample_{:09d}".format(sample_index)))

    tree = MeshToCGNS(rec_mesh)

    new_sample = Sample()
    new_sample.add_tree(tree)

    if sample_index in ids_train:
        scalar_names = in_scalars_names + out_scalars_names
        for fn in out_fields_names:
            new_sample.add_field(fn, sample.get_field(fn, base_name="Base_2_2"))
    elif sample_index in ids_test:
        scalar_names = in_scalars_names
    else:
        raise("unkown sample_index")

    for sn in scalar_names:
        new_sample.scalars.add(sn, sample.scalars.get(sn))

    new_sample.add_field("Signed_Distance", sample.get_field("sdf", base_name="Base_2_2"))

    path = os.path.join(prepared_data_dir,"dataset/samples/sample_{:09d}".format(sample_index))
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    new_sample.save(path)

print("duration prepare =", time.time()-start)
# 47 seconds