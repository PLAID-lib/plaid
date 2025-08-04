from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.problem_definition import ProblemDefinition
from Muscat.Bridges.CGNSBridge import CGNSToMesh
import numpy as np
from Muscat.Containers.Filters.FilterObjects import ElementFilter
from Muscat.Containers.MeshFieldOperations import GetFieldTransferOp
from Muscat.FE.Fields.FEField import FEField
from Muscat.Bridges.CGNSBridge import CGNSToMesh
from Muscat.FE.FETools import PrepareFEComputation
from tqdm import tqdm

import os, pickle, time, shutil, copy

start = time.time()


plaid_location = # path to update
predicted_data_dir = # path to update



datapath=os.path.join(plaid_location, "dataset")
pb_defpath=os.path.join(plaid_location, "problem_definition")



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


out_fields_names = ['mach', 'nut']
out_scalars_names = ['Q', 'power', 'Pr', 'Tr', 'eth_is', 'angle_out']
nbe_features = len(out_fields_names) + len(out_scalars_names)

rec_mesh = CGNSToMesh(dataset_pred[ids_test[0]].get_mesh())


prediction = []

count = 0
for sample_index in tqdm(ids_test):

    sample_pred = dataset_pred[sample_index]
    sample = dataset[sample_index]

    prediction.append({})
    for fn in out_fields_names:
        prediction[count][fn] = sample_pred.get_field(fn, base_name="Base_2_2")
    for sn in out_scalars_names:
        prediction[count][sn] = sample_pred.get_scalar(sn)

    count += 1


with open('prediction_vki.pkl', 'wb') as file:
    pickle.dump(prediction, file)

print("duration construct predictions =", time.time()-start)
# 15.2 seconds