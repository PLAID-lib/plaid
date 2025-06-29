from datasets import load_dataset
from plaid.containers.sample import Sample
from plaid.containers.dataset import Dataset
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

from ml_pipeline_HF_nodes import ScalarScalerNode, GPRegressorNode


with open("config_2.yml") as f:
    params = yaml.safe_load(f)


class HFDataset(Dataset):
    def __init__(self, hf_dataset = None, path = None):
        assert not (hf_dataset and path), "hf_dataset and path cannot be both initialized"
        assert hf_dataset or path, "hf_dataset and path cannot be both not initialized"
        if hf_dataset:
            self.ds = hf_dataset
        elif path:
            self.ds = load_dataset(path, split="all_samples")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return HFDataset(self.ds[idx])
        return Sample.model_validate(pickle.loads(self.ds[idx]["sample"]))

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getattr__(self, name):
        attr = getattr(self.ds, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, type(self.ds)):
                    return HFDataset(result)
                return result
            return wrapper
        return attr


start = time.time()
dataset = HFDataset(path = params['dataset_path'])
print(f"Loading dataset from HuggingFace Hub took: {time.time() - start:.2g} seconds")

params['prob_def'] = huggingface_description_to_problem_definition(dataset.description)


pipeline = Pipeline([
    ('scalar_scaler', ScalarScalerNode(name = 'scalar_scaler', params = params)),
    # ('tabular_regressor', GPRegressorNode(name = 'tabular_regressor', params = params))
])

print(dataset[0].get_scalar('max_U2_top'))


pipeline.fit_transform(dataset)
print("pipeline fitted")

print(dataset[0].get_scalar('max_U2_top'))

# dataset_2 = pipeline.inverse_transform(pipeline.predict(dataset))

# print(dataset_2[0].get_scalar('max_U2_top'))

# dataset_3 = pipeline.inverse_transform(dataset_2)

# print(dataset_3[0].get_scalar('p1'))



# dataset_4 = pipeline.inverse_transform(pipeline.fit_transform(dataset))
# print(dataset_4[0].get_scalar('p1'))
