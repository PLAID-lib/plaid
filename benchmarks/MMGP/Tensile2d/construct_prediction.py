from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition

from tqdm import tqdm

import os, pickle, time

start = time.time()


plaid_location = # to update

pb_defpath=os.path.join(plaid_location, "problem_definition")

predicted_data_dir= "data/Tensile2d_predicted/dataset"

dataset_pred = Dataset()
dataset_pred._load_from_dir_(predicted_data_dir, verbose=True, processes_number=4)

problem = ProblemDefinition()
problem._load_from_dir_(pb_defpath)

ids_train = problem.get_split('train_500')
ids_test  = problem.get_split('test')


n_train = len(ids_train)
n_test  = len(ids_test)

out_fields_names = ['U1', 'U2', 'sig11', 'sig22', 'sig12']
out_scalars_names = ['max_von_mises', 'max_U2_top', 'max_sig22_top']
nbe_features = len(out_fields_names) + len(out_scalars_names)


prediction = []

count = 0
for sample_index in tqdm(ids_test):

    sample_pred = dataset_pred[sample_index]

    prediction.append({})
    for fn in out_fields_names:
        prediction[count][fn] = sample_pred.get_field(fn+"_predicted")
    for sn in out_scalars_names:
        prediction[count][sn] = sample_pred.get_scalar(sn+"_predicted")

    count += 1

with open('prediction_tensile2d.pkl', 'wb') as file:
    pickle.dump(prediction, file)

print("duration construct predictions =", time.time()-start)
# 4 seconds

# preprocess done in 112.36455297470093 s
# train done in 1396.3768372535706 s
# inference done in 86.41910243034363 s