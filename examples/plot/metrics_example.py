from plaid.containers.dataset import Dataset
from plaid.plot.metrics import compute_metrics
from plaid.problem_definition import ProblemDefinition
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

### Metrics with file paths ###
print("Metrics with file paths")
ref_ds = os.path.join(current_directory, "dataset_ref")
pred_ds = os.path.join(current_directory, "dataset_near_pred")
problem = os.path.join(current_directory, "problem_definition")
compute_metrics(ref_ds, pred_ds, problem, "first_metrics")

### Metrics with PLAID objects and verbose ###
print("Metrics with PLAID objects and verbose")
ref_ds = Dataset(os.path.join(current_directory, "dataset_ref"))
pred_ds = Dataset(os.path.join(current_directory, "dataset_pred"))
problem = ProblemDefinition(os.path.join(current_directory, "problem_definition"))
compute_metrics(ref_ds, pred_ds, problem, "second_metrics", verbose=True)
