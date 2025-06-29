from datasets import load_dataset
from plaid.containers.sample import Sample
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition
import os, pickle
import numpy as np
from safetensors.numpy import save_file
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import joblib

import yaml
import time

from ml_pipeline_nodes import ScalarScalerNode, GPRegressorNode, PCAEmbeddingNode #, TutteMorphing


with open("config_2.yml") as f:
    config = yaml.safe_load(f)

global_params = config["global"]


start = time.time()
hf_dataset = load_dataset(global_params['dataset_path'], split="all_samples")
print(f"Loading dataset from HuggingFace Hub took: {time.time() - start:.2g} seconds")

prob_def = huggingface_description_to_problem_definition(hf_dataset.description)

ref_split = prob_def.get_split(global_params['train_split_name'])[:20]
train_split = prob_def.get_split(global_params['train_split_name'])[10:20]
test_split = prob_def.get_split(global_params['train_split_name'])[:20]

dataset_ref, _ = huggingface_dataset_to_plaid(hf_dataset, ids = ref_split)
dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, ids = train_split)
dataset_test, _ = huggingface_dataset_to_plaid(hf_dataset, ids = test_split)




pipeline = Pipeline([
    ('input_scalar_scaler', ScalarScalerNode(name = 'input_scalar_scaler', global_params = global_params, params = config['input_scalar_scaler'])),
    ('output_scalar_scaler', ScalarScalerNode(name = 'output_scalar_scaler', global_params = global_params, params = config['output_scalar_scaler'])),
    ('pca_shape_embedding', PCAEmbeddingNode(name = 'pca_shape_embedding', global_params = global_params, params = config['pca_shape_embedding'])),
    ('pca_field_embedding', PCAEmbeddingNode(name = 'pca_field_embedding', global_params = global_params, params = config['pca_field_embedding'])),
    ('tabular_regressor', GPRegressorNode(name = 'tabular_regressor', global_params = global_params, params = config['tabular_regressor']))
])

print("pipeline parameters=", pipeline.get_params(deep=True))

1./0.

ind = train_split[0]

pipeline.fit(dataset_train)
print("pipeline fitted")

dataset_ref = pipeline.transform(dataset_ref)

dataset_test_2 = pipeline.inverse_transform(pipeline.predict(dataset_ref))

print("score =", pipeline.score(dataset_test_2, dataset_ref))

print(dataset_test_2)

dataset_ref._save_to_dir_(os.path.join(params['save_path'], "dataset_ref"), verbose = True)
dataset_test_2._save_to_dir_(os.path.join(params['save_path'], "dataset_test_2"), verbose = True)


# print(dataset_train[ind])
# print(dataset_train[ind].get_scalar_names())

1./0.
dataset_2 = pipeline.inverse_transform(pipeline.predict(dataset))

print(dataset_2[0].get_scalar('max_U2_top'))

# dataset_3 = pipeline.inverse_transform(dataset_2)

# print(dataset_3[0].get_scalar('p1'))



# dataset_4 = pipeline.inverse_transform(pipeline.fit_transform(dataset))
# print(dataset_4[0].get_scalar('p1'))


test_split = prob_def.get_split(params['test_split_name'])
dataset_test, _ = huggingface_dataset_to_plaid(hf_dataset, ids = test_split)
