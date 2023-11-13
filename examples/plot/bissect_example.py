from plaid.plot.bissect import plot_bissect
from plaid.containers.dataset import Dataset
from plaid.problem_definition import ProblemDefinition


### Plot with file paths ###
print("Plot with file paths")
ref_path = "examples/plot/dataset_ref"
pred_path = "examples/plot/dataset_pred"
problem_path = "examples/plot/problem_definition"
plot_bissect(ref_path, pred_path, problem_path, "scalar_2", "differ_bissect_plot")

### Plot with PLAID objects ###
print("Plot with PLAID objects")
ref_path = Dataset("examples/plot/dataset_pred")
pred_path = Dataset("examples/plot/dataset_pred")
problem_path = ProblemDefinition("examples/plot/problem_definition")
plot_bissect(ref_path, pred_path, problem_path, "scalar_2", "equal_bissect_plot")

### Mix with scalar index and verbose ###
print("Mix with scalar index and verbose")
scalar_index = 0
ref_path = "examples/plot/dataset_ref"
pred_path = "examples/plot/dataset_near_pred"
problem_path = ProblemDefinition("examples/plot/problem_definition")
plot_bissect(ref_path, pred_path, problem_path, scalar_index, "converge_bissect_plot", verbose=True)