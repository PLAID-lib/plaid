# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import os
import yaml
import time

from datasets import load_dataset
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, ids = train_split, processes_number = os.cpu_count())

# Pipeline with ``n_components`` specified in PCAEmbeddingNode for GridSearchCV
pipeline = Pipeline([
    ('input_scalar_scaler', ScalarScalerNode(params = config['input_scalar_scaler'])),
    ('output_scalar_scaler', ScalarScalerNode(params = config['output_scalar_scaler'])),
    ('pca_nodes', PCAEmbeddingNode(params = config['pca_nodes'], n_components = config['pca_nodes']['n_components'])),
    ('pca_mach', PCAEmbeddingNode(params = config['pca_mach'], n_components = config['pca_mach']['n_components'])),
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
test_split = prob_def.get_split(global_params['train_split_name'])[:24]

dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, ids = train_split, processes_number = os.cpu_count())
dataset_test, _ = huggingface_dataset_to_plaid(hf_dataset, ids = test_split, processes_number = os.cpu_count())

# Pipeline with all arguments specified in ``config.yml``
pipeline = Pipeline([
    ('input_scalar_scaler', ScalarScalerNode(params = config['input_scalar_scaler'])),
    ('output_scalar_scaler', ScalarScalerNode(params = config['output_scalar_scaler'])),
    ('pca_nodes', PCAEmbeddingNode(params = config['pca_nodes'])),
    ('pca_mach', PCAEmbeddingNode(params = config['pca_mach'])),
    ('regressor_mach', GPRegressorNode(params = config['regressor_mach']))
])


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
