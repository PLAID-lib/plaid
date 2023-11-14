import numpy as np
import yaml
from sklearn.metrics import r2_score
from tqdm import tqdm

from plaid.containers.dataset import Dataset
from plaid.plot.bissect import prepare_datasets
from plaid.problem_definition import ProblemDefinition


def compute_rRMSE_RMSE(metrics: dict, rel_SE_out_scalars: dict, abs_SE_out_scalars: dict,
                       problem_split: dict, out_scalars_names: list[str], verbose: bool) -> None:
    """Compute and print the relative Root Mean Square Error (rRMSE) for scalar outputs.

    Args:
        metrics (dict): Dictionary to store the computed metrics.
        rel_SE_out_scalars (dict): Dictionary containing relative squared errors for scalar outputs.
        problem_split (dict): Dictionary specifying how the problem is split.
        out_scalars_names (list[str]): List of names of scalar outputs.
        verbose (bool): If True, print detailed information during computation.
    """
    metrics["rRMSE for scalars"] = {}
    metrics["RMSE for scalars"] = {}

    for split_name, _ in problem_split.items():
        metrics["rRMSE for scalars"][split_name] = {}
        metrics["RMSE for scalars"][split_name] = {}

        for sname in out_scalars_names:
            rRMSE_value = np.sqrt(
                np.mean(rel_SE_out_scalars[split_name][sname], axis=0))
            out_string_rRMSE = "{:#.6g}".format(rRMSE_value)

            RMSE_value = np.sqrt(
                np.mean(abs_SE_out_scalars[split_name][sname], axis=0))
            out_string_RMSE = "{:#.6g}".format(RMSE_value)

            metrics["rRMSE for scalars"][split_name][sname] = float(out_string_rRMSE)
            metrics["RMSE for scalars"][split_name][sname] = float(out_string_RMSE)


def compute_R2(metrics: dict, r2_out_scalars: dict, problem_split: dict,
               out_scalars_names: list[str], verbose: bool) -> None:
    """Compute and print the R-squared (R2) score for scalar outputs.

    Args:
        metrics (dict): Dictionary to store the computed metrics.
        r2OutScalars (dict): Dictionary containing R2 scores for scalar outputs.
        problem_split (dict): Dictionary specifying how the problem is split.
        out_scalars_names (list[str]): List of names of scalar outputs.
        verbose (bool): If True, print detailed information during computation.
    """
    metrics["R2 for scalars"] = {}

    for split_name, _ in problem_split.items():
        metrics["R2 for scalars"][split_name] = {}

        for sname in out_scalars_names:
            out_string = "{:#.6g}".format(r2_out_scalars[split_name][sname])
            metrics["R2 for scalars"][split_name][sname] = float(out_string)


def prepare_metrics_for_split(ref_out_specific_scalars: np.ndarray, pred_out_specific_scalars: np.ndarray,
                              split_indices: list[int], rel_SE_out_specific_scalars: np.ndarray,
                              abs_SE_out_specific_scalars: np.ndarray, tolerance: float = 1.e-6) -> float:
    """Prepare metrics for a specific split and compute the R-squared (R2) score.

    Args:
        ref_out_specific_scalars (np.ndarray): Array of reference scalar outputs.
        pred_out_specific_scalars (np.ndarray): Array of predicted scalar outputs.
        split_indices (list[int]): List of indices specifying the split.
        rel_SE_out_specific_scalars (np.ndarray): Array to store relative squared errors for scalar outputs.
        tolerance (float, optional): Tolerance value for denominator. Defaults to 1.e-6.

    Returns:
        float: R-squared (R2) score for the specific split.
    """
    ref_scal = np.array([ref_out_specific_scalars[i] for i in split_indices])
    predict_scal = np.array([pred_out_specific_scalars[i]
                            for i in split_indices])

    denom_scal = np.where(ref_scal < tolerance, 1., ref_scal)

    diff = (predict_scal - ref_scal)
    rel_SE_out_specific_scalars[:] = (diff / denom_scal) ** 2
    abs_SE_out_specific_scalars[:] = diff ** 2
    return r2_score(ref_scal, predict_scal)


def pretty_metrics(metrics: dict):
    """Prints metrics information in a readable format (pretty print).

    Args:
        metrics (dict): The metrics dictionary to print.
    """
    metrics_keys = list(metrics.keys())
    from Muscat.Helpers.TextFormatHelper import TFormat
    tf = TFormat.Center(TFormat.InBlue("comparision metrics")) + '\n'
    for metric_key in metrics_keys:
        tf += TFormat.GetIndent() + TFormat.InYellow(metric_key) + '\n'
        splits = list(metrics[metric_key].keys())
        TFormat.II()

        for split in splits:
            tf += TFormat.GetIndent() + TFormat.InGreen(split) + '\n'
            scalars = list(metrics[metric_key][split].keys())
            TFormat.II()

            for scalar in scalars:
                tf += TFormat.GetIndent() + TFormat.InBlue(scalar) + ": " + \
                    str(metrics[metric_key][split][scalar]) + '\n'

            TFormat.DI()
        TFormat.DI()
    tf += TFormat.Center('') + '\n'
    print(tf)


def compute_metrics(ref_dataset: Dataset | str, pred_dataset: Dataset | str, problem: ProblemDefinition |
                    str, save_file_name: str = "test_metrics", verbose: bool = False) -> None:
    """Compute and save evaluation metrics for a given regression problem.

    Args:
        ref_dataset (Dataset | str): Reference dataset or path to a reference dataset.
        pred_dataset (Dataset | str): Predicted dataset or path to a predicted dataset.
        problem (ProblemDefinition | str): Problem definition or path to a problem definition.
        save_file_name (str, optional): Name of the file to save the metrics. Defaults to "test_metrics".
        verbose (bool, optional): If True, print detailed information during computation.
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
    abs_SE_out_scalars = {}
    r2_out_scalars = {}

    for split_name, split_indices in problem_split.items():
        rel_SE_out_scalars[split_name] = {}
        abs_SE_out_scalars[split_name] = {}
        r2_out_scalars[split_name] = {}
        for sname in out_scalars_names:
            rel_SE_out_scalars[split_name][sname] = np.empty(
                len(split_indices))
            abs_SE_out_scalars[split_name][sname] = np.empty(
                len(split_indices))
            r2_out_scalars[split_name][sname] = np.empty(1)

    print("Compute metrics for each regressor:") if verbose else None
    for split_name, split_indices in tqdm(
            problem_split.items(), disable=not (verbose)):
        for sname in out_scalars_names:
            r2_out_scalars[split_name][sname] = prepare_metrics_for_split(
                ref_out_scalars[sname], pred_out_scalars[sname], split_indices,
                rel_SE_out_scalars[split_name][sname], abs_SE_out_scalars[split_name][sname]
            )

    metrics = {}
    compute_rRMSE_RMSE(
        metrics,
        rel_SE_out_scalars,
        abs_SE_out_scalars,
        problem_split,
        out_scalars_names,
        verbose)
    compute_R2(
        metrics,
        r2_out_scalars,
        problem_split,
        out_scalars_names,
        verbose)

    with open(f"{save_file_name}.yaml", 'w') as file:
        yaml.dump(metrics, file, default_flow_style=False, sort_keys=False)

    if verbose:
        pretty_metrics(metrics)

    return metrics