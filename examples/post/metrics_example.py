from plaid.containers.dataset import Dataset
from plaid.post.metrics import compute_metrics
from plaid.problem_definition import ProblemDefinition
import os, shutil

current_directory = os.path.dirname(os.path.abspath(__file__))
working_directory = os.getcwd()

### Metrics with file paths ###
print("Metrics with file paths")
ref_ds = os.path.join(current_directory, "dataset_ref")
pred_ds = os.path.join(current_directory, "dataset_near_pred")
problem = os.path.join(current_directory, "problem_definition")
compute_metrics(ref_ds, pred_ds, problem, "first_metrics")
shutil.move(os.path.join(working_directory, "first_metrics.yaml"), os.path.join(current_directory, "first_metrics.yaml"))

### Metrics with PLAID objects and verbose ###
print("Metrics with PLAID objects and verbose")
ref_ds = Dataset(os.path.join(current_directory, "dataset_ref"))
pred_ds = Dataset(os.path.join(current_directory, "dataset_pred"))
problem = ProblemDefinition(os.path.join(current_directory, "problem_definition"))
compute_metrics(ref_ds, pred_ds, problem, "second_metrics", verbose=True)
shutil.move(os.path.join(working_directory, "second_metrics.yaml"), os.path.join(current_directory, "second_metrics.yaml"))
