from plaid.containers.dataset import Dataset
from plaid.post.bissect import plot_bissect
from plaid.problem_definition import ProblemDefinition
import os, shutil

current_directory = os.path.dirname(os.path.abspath(__file__))
working_directory = os.getcwd()

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
shutil.move(os.path.join(working_directory, "differ_bissect_plot.png"), os.path.join(current_directory, "differ_bissect_plot.png"))

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
shutil.move(os.path.join(working_directory, "equal_bissect_plot.png"), os.path.join(current_directory, "equal_bissect_plot.png"))

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
shutil.move(os.path.join(working_directory, "converge_bissect_plot.png"), os.path.join(current_directory, "converge_bissect_plot.png"))
