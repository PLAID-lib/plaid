import numpy as np
import yaml

from tqdm import tqdm
from sklearn.metrics import r2_score
from plaid.plot.bissect import prepare_datasets
from plaid.problem_definition import ProblemDefinition
from plaid.containers.dataset import Dataset

def compute_rRMSE(metrics, rel_SE_out_scalars, problem_split, out_scalars_names, number_of_regressors, verbose):
    metrics["rRMSE for scalars"] = {}
    print("===\nrRMSE for scalars") if verbose else None

    for split_name, _ in problem_split.items():
        metrics["rRMSE for scalars"][split_name] = {}
        print("  " + split_name) if verbose else None

        for sname in out_scalars_names:
            rel_RMSE_out_scalars_set = np.sqrt(np.mean(rel_SE_out_scalars[split_name][sname], axis=1))

            out_string = "{:#.6g}".format(np.mean(rel_RMSE_out_scalars_set))
            out_string += " +/- {:#.6g}".format(np.std(rel_RMSE_out_scalars_set))

            metrics["rRMSE for scalars"][split_name][sname] = out_string
            print(sname.ljust(14) + ": " + out_string) if verbose else None


def compute_R2(metrics, r2OutScalars, problem_split, out_scalars_names, number_of_regressors, verbose):
    metrics["R2 for scalars"] = {}
    print("===\nR2 for scalars") if verbose else None

    for split_name, _ in problem_split.items():
        metrics["R2 for scalars"][split_name] = {}
        print("  " + split_name) if verbose else None

        for sname in out_scalars_names:
            out_string = "{:#.6g}".format(np.mean(r2OutScalars[split_name][sname]))
            out_string += " +/- {:#.6g}".format(np.std(r2OutScalars[split_name][sname]))

            metrics["R2 for scalars"][split_name][sname] = out_string
            print(sname.ljust(14) + ": " + out_string) if verbose else None


def plot_metrics(ref_dataset: Dataset | str, pred_dataset: Dataset | str,
                 problem: ProblemDefinition | str, save_file_name: str = "test_metrics",
                 number_of_regressors: int = 10, verbose: bool = True):
    """_summary_

    Args:
        ref_dataset (Dataset | str): _description_
        pred_dataset (Dataset | str): _description_
        problem (ProblemDefinition | str): _description_
        save_file_name (str, optional): _description_. Defaults to "test_metrics".
        number_of_regressors (int, optional): _description_. Defaults to 10.
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    ### Transform path to Dataset object ###
    if isinstance(ref_dataset, str):
        ref_dataset: Dataset = Dataset(ref_dataset)
    if isinstance(pred_dataset, str):
        pred_dataset: Dataset = Dataset(pred_dataset)
    if isinstance(problem, str):
        problem: ProblemDefinition = ProblemDefinition(problem)

    ### Get important formated values ###
    problem_split = problem.get_split()
    ref_out_scalars, pred_out_scalars, out_scalars_names = prepare_datasets(
        ref_dataset, pred_dataset, problem, verbose)


    rel_SE_out_scalars = {}
    r2OutScalars = {}

    tolerance = 1.e-6

    for split_name, split_indices in problem_split.items():
        rel_SE_out_scalars[split_name] = {}
        r2OutScalars[split_name] = {}
        for sname in out_scalars_names:
            rel_SE_out_scalars[split_name][sname] = np.empty(
                (number_of_regressors, len(split_indices)))
            r2OutScalars[split_name][sname] = np.empty(number_of_regressors)

    print("Compute metrics for each regressor:") if verbose else None

    for J in tqdm(range(number_of_regressors), disable=not (verbose)):
        for split_name, split_indices in problem_split.items():
            for sname in out_scalars_names:

                ref_scal = np.array([])
                predict_scal = np.array([])

                for i, index in enumerate(split_indices):

                    ref  = ref_out_scalars[sname][index]
                    pred = pred_out_scalars[sname][index]

                    ref_scal = np.hstack((ref_scal, ref))
                    predict_scal = np.hstack((predict_scal, pred))

                    # Compute relative difference
                    if ref < tolerance:
                        denom_scal = 1.
                    else:
                        denom_scal = ref
                    reldif = (pred - ref) / denom_scal
                    rel_SE_out_scalars[split_name][sname][J, i] = reldif**2

                r2OutScalars[split_name][sname][J] = r2_score(
                    ref_scal, predict_scal)

    metrics = {}
    compute_rRMSE(metrics, rel_SE_out_scalars, problem_split, out_scalars_names, number_of_regressors, verbose)
    compute_R2(metrics, r2OutScalars, problem_split, out_scalars_names, number_of_regressors, verbose)

    with open(f"{save_file_name}.yaml", 'w') as file:
        yaml.dump(metrics, file, default_flow_style=False, sort_keys=False)

    return metrics

ref_ds = Dataset("examples/plot/dataset_ref")
pred_ds = Dataset("examples/plot/dataset_near_pred")
problem = ProblemDefinition("examples/plot/problem_definition")
plot_metrics(ref_ds, pred_ds, problem)

ref_ds = Dataset("examples/plot/dataset_ref")
pred_ds = Dataset("examples/plot/dataset_pred")
problem = ProblemDefinition("examples/plot/problem_definition")
plot_metrics(ref_ds, pred_ds, problem)