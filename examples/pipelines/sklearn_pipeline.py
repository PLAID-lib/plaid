from datasets import load_dataset
from plaid.containers.sample import Sample
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition
import os, pickle
import numpy as np
from safetensors.numpy import save_file
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import joblib
from copy import copy

import yaml
import time

from ml_pipeline_nodes import ScalarScalerNode, GPRegressorNode, PCAEmbeddingNode

import warnings
warnings.filterwarnings('ignore', module='sklearn')


with open("config.yml") as f:
    config = yaml.safe_load(f)

global_params = config["global"]


start = time.time()
hf_dataset = load_dataset(global_params['dataset_path'], split="all_samples")
print(f"Loading dataset from HuggingFace Hub took: {time.time() - start:.2g} seconds")

prob_def = huggingface_description_to_problem_definition(hf_dataset.description)

train_split = prob_def.get_split(global_params['train_split_name'])[:100]
dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, ids = train_split)


pipeline = Pipeline([
    ('input_scalar_scaler', ScalarScalerNode(type = config['input_scalar_scaler']['type'], scalar_names = config['input_scalar_scaler']['scalar_names'])),
    ('output_scalar_scaler', ScalarScalerNode(type = config['output_scalar_scaler']['type'], scalar_names = config['output_scalar_scaler']['scalar_names'])),
    ('pca_nodes', PCAEmbeddingNode(field_name = config['pca_nodes']['field_name'], n_components = config['pca_nodes']['n_components'], base_name = config['pca_nodes']['base_name'])),
    ('pca_mach', PCAEmbeddingNode(field_name = config['pca_mach']['field_name'], n_components = config['pca_mach']['n_components'], base_name = config['pca_mach']['base_name'])),
    ('regressor_mach', GPRegressorNode(params = config['regressor_mach']))
])



print("=================================")
print("GridSearchCV example:")
# print("pipeline parameters=", pipeline.get_params(deep=True))
# print("pipeline parameters=", pipeline.get_params().keys())
# print("Pipeline steps:", pipeline.steps)


param_grid = {
    'pca_nodes__n_components': [2, 3],
    'pca_mach__n_components': [4, 5],
}

# Run GridSearchCV
search = GridSearchCV(pipeline, param_grid=param_grid, cv=3, verbose=3)
search.fit(dataset_train)

# Results
print("Best parameters:", search.best_params_)
print("Best score:", search.best_score_)



print("=================================")
print("Direct pipeline example:")

train_split = prob_def.get_split(global_params['train_split_name'])
test_split = prob_def.get_split(global_params['train_split_name'])[:10]

dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, ids = train_split)
dataset_test, _ = huggingface_dataset_to_plaid(hf_dataset, ids = test_split)



pipeline.fit(dataset_train)
print("pipeline fitted")

dataset_test_transformed = pipeline.transform(dataset_test)

print("score =", pipeline.score(dataset_test, dataset_test_transformed))


dataset_test_pred = pipeline.inverse_transform(pipeline.predict(dataset_test))

import shutil

if os.path.exists(os.path.join(global_params['save_path'], "dataset_test")):
    shutil.rmtree(os.path.join(global_params['save_path'], "dataset_test"))
dataset_test._save_to_dir_(os.path.join(global_params['save_path'], "dataset_test"), verbose = True)

if os.path.exists(os.path.join(global_params['save_path'], "dataset_test_pred")):
    shutil.rmtree(os.path.join(global_params['save_path'], "dataset_test_pred"))
dataset_test_pred._save_to_dir_(os.path.join(global_params['save_path'], "dataset_test_pred"), verbose = True)
