from datasets import load_from_disk, load_dataset
from plaid.containers.sample import Sample
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition
import os, pickle
import numpy as np
from safetensors.numpy import save_file
from sklearn.base import BaseEstimator, RegressorMixin

from pipefunc import Pipeline, pipefunc
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import joblib

import yaml
import time



@pipefunc(output_name=("dataset", "prob_def"))
def load_hf_from_disk(path):
    return huggingface_dataset_to_plaid(load_from_disk(path))


@pipefunc(output_name=("dataset", "prob_def"))
def load_hf_from_hub(path):
    start = time.time()
    hf_dataset = load_dataset(path, split="all_samples")
    print(f"Loading dataset from HuggingFace Hub took: {time.time() - start:.2g} seconds")
    dataset = huggingface_dataset_to_plaid(hf_dataset)
    return dataset


@pipefunc(output_name=("scalar_data"))
def scale_scalars(dataset, prob_def, train_split_name, test_split_name, out_path):

    ids_train = prob_def.get_split(train_split_name)
    input_scalars_train = dataset.get_scalars_to_tabular(
        scalar_names = prob_def.get_input_scalars_names(),
        sample_ids = ids_train,
        as_nparray = True
    )
    output_scalars_train = dataset.get_scalars_to_tabular(
        scalar_names = prob_def.get_output_scalars_names(),
        sample_ids = ids_train,
        as_nparray = True
    )

    ids_test = prob_def.get_split(test_split_name)
    input_scalars_test = dataset.get_scalars_to_tabular(
        scalar_names = prob_def.get_input_scalars_names(),
        sample_ids = ids_test,
        as_nparray = True
    )

    input_scalar_scaler = StandardScaler()
    input_scalars_train = input_scalar_scaler.fit_transform(input_scalars_train)
    input_scalars_test = input_scalar_scaler.transform(input_scalars_test)

    output_scalar_scaler = StandardScaler()
    output_scalars_train = output_scalar_scaler.fit_transform(output_scalars_train)

    scalar_data = [
        input_scalar_scaler,
        output_scalar_scaler,
        input_scalars_train,
        input_scalars_test,
        output_scalars_train
    ]

    os.makedirs(out_path, exist_ok=True)

    tensors = {
        "scaler_mean": input_scalar_scaler.mean_,
        "scaler_scale": input_scalar_scaler.scale_,
    }
    saved_path = os.path.join(out_path, f"input_scalar_scaler.safetensors")
    save_file(tensors, saved_path)

    tensors = {
        "scaler_mean": output_scalar_scaler.mean_,
        "scaler_scale": output_scalar_scaler.scale_,
    }
    saved_path = os.path.join(out_path, f"output_scalar_scaler.safetensors")
    save_file(tensors, saved_path)

    return scalar_data


def extract_leaf_keys(d):
    leaves = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict) or isinstance(v, list):
                leaves.update(extract_leaf_keys(v))
            else:
                leaves[k] = v
    elif isinstance(d, list):
        for item in d:
            leaves.update(extract_leaf_keys(item))
    return leaves


if __name__ == "__main__":

    start = time.time()

    pipeline = Pipeline(
            [
                load_hf_from_hub,
                scale_scalars,
            ],
        name="ML_Workflow",
        profile=True
    )

    # pipeline.visualize()

    with open("config.yml") as f:
        config = yaml.safe_load(f)

    parameters = extract_leaf_keys(config)

    scalar_data = pipeline(**parameters)

    print(f"Pipeline execution time {time.time() - start:.2g} seconds")


    pipeline.print_profiling_stats()
    # print("Dataset:", type(dataset[0:10]), type(dataset))

    # print(scalar_data)
