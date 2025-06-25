from datasets import load_from_disk
from plaid.containers.sample import Sample
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition
import os, pickle
from safetensors.numpy import save_file

from pipefunc import Pipeline, pipefunc
from sklearn.preprocessing import StandardScaler

import yaml


@pipefunc(output_name=("dataset", "prob_def"))
def load_hf_from_disk(path):
    return huggingface_dataset_to_plaid(load_from_disk(path))

@pipefunc(output_name="scalar_scalers")
def scale_scalars(dataset, prob_def, train_split_name, test_split_name):

    print(">>", dataset[0].get_scalar("p1"))

    ids_train = prob_def.get_split(train_split_name)
    train_scalars = dataset.get_scalars_to_tabular(
        sample_ids = ids_train
    )

    ids_test = prob_def.get_split(test_split_name)
    test_scalars = dataset.get_scalars_to_tabular(
        sample_ids = ids_test
    )

    scalar_scalers = {}

    print(prob_def.get_input_scalars_names())

    for sn in prob_def.get_input_scalars_names():
        scaler = StandardScaler()
        scaler.fit_transform(train_scalars[sn].reshape(-1, 1))
        scaler.transform(test_scalars[sn].reshape(-1, 1))
        scalar_scalers[sn] = scaler

    for sn in prob_def.get_output_scalars_names():
        scaler = StandardScaler()
        scaler.fit_transform(train_scalars[sn].reshape(-1, 1))
        scalar_scalers[sn] = scaler

    for j, i in enumerate(ids_train):
        sample = dataset[i]
        for sn, scaler in scalar_scalers.items():
            if sn in sample.get_scalar_names():
                sample.add_scalar(sn, train_scalars[sn][j])

    for j, i in enumerate(ids_test):
        sample = dataset[i]
        for sn, scaler in scalar_scalers.items():
            if sn in sample.get_scalar_names():
                sample.add_scalar(sn, test_scalars[sn][j])


    print(">>", dataset[0].get_scalar("p1"))

    return scalar_scalers

@pipefunc(output_name="saved_path")
def save(scalar_scalers, out_path, dataset):

    os.makedirs(out_path, exist_ok=True)

    # Save only NumPy-compatible parameters
    for sn, scaler in scalar_scalers.items():
        tensors = {
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        }
        saved_path = os.path.join(out_path, f"scaler_{sn}.safetensors")
        save_file(tensors, saved_path)

    return dataset


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
    # Create the pipeline
    pipeline = Pipeline(
        [
            load_hf_from_disk,
            scale_scalars,
            save,
        ],
        profile=True
    )

    # pipeline.visualize()

    with open("config.yml") as f:
        config = yaml.safe_load(f)

    parameters = extract_leaf_keys(config)
    print(parameters)


    # Run the pipeline
    dataset = pipeline(**parameters)
    # pipeline.print_profiling_stats()
    # print("Dataset:", type(dataset[0:10]), type(dataset))

    print(dataset)

    print(">>", dataset[0].get_scalar("p1"))

