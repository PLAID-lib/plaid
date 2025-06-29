import numpy as np
from skimage import data, filters, measure
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries

from pipefunc import Pipeline, pipefunc


# Step 1: Image Loading and Preprocessing
@pipefunc(output_name="gray_image", mapspec="image[n] -> gray_image[n]")
def load_and_preprocess_image(image):
    return rgb2gray(image)


# Step 2: Image Segmentation
@pipefunc(output_name="segmented_image", mapspec="gray_image[n] -> segmented_image[n]")
def segment_image(gray_image):
    return filters.sobel(gray_image)


# Step 3: Feature Extraction
@pipefunc(output_name="feature", mapspec="segmented_image[n] -> feature[n]")
def extract_feature(segmented_image):
    boundaries = find_boundaries(segmented_image > 0.1)
    labeled_image = measure.label(boundaries)
    num_regions = np.max(labeled_image)
    return {"num_regions": num_regions}


# Step 4: Object Classification
@pipefunc(output_name="classification", mapspec="feature[n] -> classification[n]")
def classify_object(feature):
    # Classify image as 'Complex' if the number of regions is above a threshold.
    classification = "Complex" if feature["num_regions"] > 5 else "Simple"
    return classification


# Step 5: Result Aggregation
@pipefunc(output_name="summary")
def aggregate_results(classification):
    simple_count = sum(1 for c in classification if c == "Simple")
    complex_count = len(classification) - simple_count
    return {"Simple": simple_count, "Complex": complex_count}


if __name__ == "__main__":
    # Create the pipeline
    pipeline_img = Pipeline(
        [
            load_and_preprocess_image,
            segment_image,
            extract_feature,
            classify_object,
            aggregate_results,
        ],
    )

    # Simulate a batch of images (using built-in scikit-image sample images)
    images = [
        data.astronaut(),
        data.coffee(),
        data.coffee(),
    ]  # Repeat the coffee image to simulate multiple images

    # Run the pipeline on the images
    results_summary = pipeline_img.map({"image": images})
    print("Classification Summary:", results_summary["summary"].output)