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
    params = yaml.safe_load(f)


start = time.time()
hf_dataset = load_dataset(params['dataset_path'], split="all_samples")
print(f"Loading dataset from HuggingFace Hub took: {time.time() - start:.2g} seconds")

prob_def = huggingface_description_to_problem_definition(hf_dataset.description)

train_split = prob_def.get_split(params['train_split_name'])[:20]
test_split = prob_def.get_split(params['train_split_name'])[:20]

dataset_train, _ = huggingface_dataset_to_plaid(hf_dataset, ids = train_split)
dataset_test, _ = huggingface_dataset_to_plaid(hf_dataset, ids = test_split)



pipeline = Pipeline([
    ('input_scalar_scaler', ScalarScalerNode(name = 'input_scalar_scaler', params = params)),
    ('output_scalar_scaler', ScalarScalerNode(name = 'output_scalar_scaler', params = params)),
    ('pca_shape_embedding', PCAEmbeddingNode(name = 'pca_shape_embedding', params = params)),
    ('pca_field_embedding', PCAEmbeddingNode(name = 'pca_field_embedding', params = params)),
    ('tabular_regressor', GPRegressorNode(name = 'tabular_regressor', params = params))
])

ind = train_split[0]

pipeline.fit(dataset_train)
print("pipeline fitted")

dataset_test_2 = pipeline.predict(dataset_test)

print("score =", pipeline.score(dataset_test, dataset_test_2))

print(dataset_test_2)

# dataset_test._save_to_dir_(os.path.join(params['save_path'], "dataset_test"), verbose = True)
# dataset_test_2._save_to_dir_(os.path.join(params['save_path'], "dataset_test_2"), verbose = True)

from Muscat.Bridges.CGNSBridge import CGNSToMesh, MeshToCGNS
from Muscat.IO.XdmfWriter import WriteMeshToXdmf

mesh1 = CGNSToMesh(dataset_test[0].get_mesh(), baseNames = ["Base_2_2"])
mesh2 = CGNSToMesh(dataset_test_2[0].get_mesh(), baseNames = ["Base_2_2"])

print(mesh1)
print(mesh2)

WriteMeshToXdmf(os.path.join(params['save_path'], "mesh1.xdmf"),
                    mesh1,
                    PointFields = [dataset_test[0].get_field(fn, base_name = "Base_2_2") for fn in ['mach', 'nut']],
                    PointFieldsNames = ['mach', 'nut']
)
WriteMeshToXdmf(os.path.join(params['save_path'], "mesh2.xdmf"),
                    mesh2,
                    PointFields = [dataset_test_2[0].get_field(fn, base_name = "Base_2_2") for fn in ['mach', 'nut']],
                    PointFieldsNames = ['mach', 'nut']
)

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
