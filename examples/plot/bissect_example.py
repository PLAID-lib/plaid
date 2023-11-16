from plaid.containers.dataset import Dataset
from plaid.plot.bissect import plot_bissect
from plaid.problem_definition import ProblemDefinition
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

### Plot with file paths ###
print("Plot with file paths")
ref_path = os.path.join(current_directory, "dataset_ref")
pred_path = os.path.join(current_directory, "dataset_pred")
problem_path = os.path.join(current_directory, "problem_definition")
plot_bissect(
    ref_path,
    pred_path,
    problem_path,
    "scalar_2",
    "differ_bissect_plot")

### Plot with PLAID objects ###
print("Plot with PLAID objects")
# Compare dataset with itself
ref_path = Dataset(os.path.join(current_directory, "dataset_pred"))
pred_path = Dataset(os.path.join(current_directory, "dataset_pred"))
problem_path = ProblemDefinition(os.path.join(current_directory, "problem_definition"))
plot_bissect(
    ref_path,
    pred_path,
    problem_path,
    "scalar_2",
    "equal_bissect_plot")

### Mix with scalar index and verbose ###
print("Mix with scalar index and verbose")
scalar_index = 0
ref_path = os.path.join(current_directory, "dataset_ref")
pred_path = os.path.join(current_directory, "dataset_near_pred")
problem_path = ProblemDefinition(os.path.join(current_directory, "problem_definition"))
plot_bissect(
    ref_path,
    pred_path,
    problem_path,
    scalar_index,
    "converge_bissect_plot",
    verbose=True)
