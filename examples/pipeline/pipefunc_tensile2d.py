from datasets import load_from_disk
from plaid.containers.sample import Sample
from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid, huggingface_description_to_problem_definition
import os, pickle
from safetensors.numpy import save_file

from pipefunc import Pipeline, pipefunc
from sklearn.preprocessing import StandardScaler



# ids_train = hf_dataset.description["split"]['train_500']
# ids_test  = hf_dataset.description["split"]['test']

# sample_train_0 = Sample.model_validate(pickle.loads(hf_dataset[ids_train[0]]["sample"]))
# sample_test_0 = Sample.model_validate(pickle.loads(hf_dataset[ids_test[0]]["sample"]))


# Step 1: Image Loading and Preprocessing
@pipefunc(output_name="hf_dataset")
def load(path):
    return load_from_disk(path)

@pipefunc(output_name="dataset")
def convert_to_plaid(hf_dataset):
    return huggingface_dataset_to_plaid(hf_dataset)[0]

@pipefunc(output_name="problem_definition")
def generate_prob_def(hf_dataset):
    return huggingface_description_to_problem_definition(hf_dataset.description)

# # Step 2: Image Segmentation
@pipefunc(output_name="scaler")
def scale_scalars(dataset, problem_definition, train_split_name, test_split_name):

    ids_train = problem_definition.get_split(train_split_name)
    train_scalars = dataset.get_scalars_to_tabular(
        sample_ids = ids_train
    )

    ids_test = problem_definition.get_split(test_split_name)
    test_scalars = dataset.get_scalars_to_tabular(
        sample_ids = ids_test
    )

    for sn in problem_definition.get_input_scalars_names():
        scaler = StandardScaler()
        scaler.fit_transform(train_scalars[sn].reshape(-1, 1))
        scaler.fit(test_scalars[sn].reshape(-1, 1))

    # print(ids_train)
    # print(ids_test)
    return scaler

@pipefunc(output_name="saved_path")
def save(scaler, out_path):

    os.makedirs(out_path, exist_ok=True)

    # Save only NumPy-compatible parameters
    tensors = {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }
    saved_path = os.path.join(out_path, "scaler.safetensors")
    save_file(tensors, saved_path)
    return saved_path

# # Step 3: Feature Extraction
# @pipefunc(output_name="feature", mapspec="segmented_image[n] -> feature[n]")
# def extract_feature(segmented_image):
#     boundaries = find_boundaries(segmented_image > 0.1)
#     labeled_image = measure.label(boundaries)
#     num_regions = np.max(labeled_image)
#     return {"num_regions": num_regions}


# # Step 4: Object Classification
# @pipefunc(output_name="classification", mapspec="feature[n] -> classification[n]")
# def classify_object(feature):
#     # Classify image as 'Complex' if the number of regions is above a threshold.
#     classification = "Complex" if feature["num_regions"] > 5 else "Simple"
#     return classification


# # Step 5: Result Aggregation
# @pipefunc(output_name="summary")
# def aggregate_results(classification):
#     simple_count = sum(1 for c in classification if c == "Simple")
#     complex_count = len(classification) - simple_count
#     return {"Simple": simple_count, "Complex": complex_count}


if __name__ == "__main__":
    # Create the pipeline
    pipeline = Pipeline(
        [
            load,
            convert_to_plaid,
            generate_prob_def,
            scale_scalars,
            save,
        ],
        profile=True
    )

    # pipeline.visualize()

    path = "Z:\\Users\\d582428\\Downloads\\Tensile2d"
    out_path = "Z:\\Users\\d582428\\Downloads\\Tensile2d\\artifacts"

    train_split_name = "train_500"
    test_split_name = "test"

    # Run the pipeline
    pipeline("saved_path",
            path = path,
            train_split_name=train_split_name,
            test_split_name=test_split_name,
            out_path = out_path)
    pipeline.print_profiling_stats()
    # print("Dataset:", type(dataset[0:10]), type(dataset))
