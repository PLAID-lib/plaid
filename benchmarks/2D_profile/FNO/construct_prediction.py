from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition
from Muscat.Bridges.CGNSBridge import CGNSToMesh
import numpy as np
from Muscat.Containers.Filters.FilterObjects import ElementFilter
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.Fields.FEField import FEField
from Muscat.Bridges.CGNSBridge import CGNSToMesh
from Muscat.FE.FETools import PrepareFEComputation
from tqdm import tqdm

import os, pickle, time, copy, shutil

start = time.time()


plaid_location = # path to update

datapath=os.path.join(plaid_location, "dataset")
pb_defpath=os.path.join(plaid_location, "problem_definition")

predicted_data_dir = # path to update


dataset_pred = Dataset()
dataset_pred._load_from_dir_(predicted_data_dir, verbose=True, processes_number=4)

problem = ProblemDefinition()
problem._load_from_dir_(pb_defpath)

ids_train = problem.get_split('train')
ids_test  = problem.get_split('test')


dataset = Dataset()
dataset._load_from_dir_(datapath, ids=ids_test, verbose=True, processes_number=4)


n_train = len(ids_train)
n_test  = len(ids_test)

out_fields_names = ['Mach', 'Pressure', 'Velocity-x', 'Velocity-y']
nbe_features = len(out_fields_names)

rec_mesh = CGNSToMesh(dataset_pred[ids_test[0]].get_mesh())


prediction = []

count = 0
for sample_index in tqdm(ids_test):

    sample_pred = dataset_pred[sample_index]
    sample = dataset[sample_index]

    input_mesh = CGNSToMesh(sample.get_mesh())

    space, numberings,_,_ = PrepareFEComputation(rec_mesh, numberOfComponents=1)
    field_mesh = FEField("", mesh=rec_mesh, space=space, numbering=numberings[0])
    efilter = ElementFilter(dimensionality=rec_mesh.GetElementsDimensionality())
    op, status, _  =  GetFieldTransferOp(inputField= field_mesh, targetPoints= input_mesh.nodes, method="Interp/Clamp" , elementFilter=efilter  , verbose=False)

    prediction.append({})
    for fn in out_fields_names:
        prediction[count][fn] = op.dot(sample_pred.get_field(fn))

    count += 1

with open('prediction_2d_profile.pkl', 'wb') as file:
    pickle.dump(prediction, file)

print("duration construct predictions =", time.time()-start)
# 24.2 seconds