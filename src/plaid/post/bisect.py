"""Utiliy function to plot bisect graphs comparing predictions vs. targets dataset."""

import subprocess
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from plaid import Dataset, ProblemDefinition


def prepare_datasets(
    ref_dataset: Dataset,
    pred_dataset: Dataset,
    problem_definition: ProblemDefinition,
    verbose: bool = False,
) -> tuple[dict, dict, list[str]]:
    """Prepare datasets for comparison.

    Args:
        ref_dataset (Dataset): The reference dataset.
        pred_dataset (Dataset): The predicted dataset.
        problem_definition (ProblemDefinition): The common problem for the reference and predicted dataset
        verbose (bool, optional): Verbose mode. Defaults to False.

    Returns:
        tuple[dict[str, list[float]], dict[str, list[float]], list[str]]: A tuple containing dictionaries of reference and predicted scalar values, and a list of scalar names.
    """
    assert len(ref_dataset) == len(pred_dataset), (
        "Reference and predicted dataset lengths differ"
    )
    ref_problem = ref_dataset.get_scalar_names()
    pred_problem = pred_dataset.get_scalar_names()
    assert ref_problem == pred_problem, "Reference and predicted dataset scalars differ"

    n_samples = len(ref_dataset)
    out_scalars_names = problem_definition.get_output_scalars_names()

    ref_out_scalars = {}
    pred_out_scalars = {}

    ref_out_scalars = {sname: [] for sname in out_scalars_names}
    pred_out_scalars = {sname: [] for sname in out_scalars_names}

    for i_sample in tqdm(range(n_samples), disable=not (verbose)):
        for sname in out_scalars_names:
            ref = ref_dataset[i_sample].get_scalar(sname)
            ref_out_scalars[sname].append(ref)

            pred = pred_dataset[i_sample].get_scalar(sname)
            pred_out_scalars[sname].append(pred)

    return ref_out_scalars, pred_out_scalars, out_scalars_names


def is_dvipng_available(verbose: bool) -> bool:
    """Check if dvipng is available on the system for matplotlib figures.

    Returns:
        bool: True if dvipng is available, False otherwise.
    """
    try:
        subprocess.run(
            ["dvipng", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True  # pragma: no cover
    except FileNotFoundError:  # pragma: no cover
        print(
            "dvipng module not installed. Using the default matplotlib options instead"
        ) if verbose else None
        return False


def plot_bisect(
    ref_dataset: Union[Dataset, str, Path],
    pred_dataset: Union[Dataset, str, Path],
    problem_def: Union[ProblemDefinition, str, Path],
    scalar: Union[str, int],
    save_file_name: str = "bissec_plots",
    verbose: bool = False,
) -> None:
    """Plot a bisect graph comparing predictions vs. targets dataset.

    Args:
        ref_dataset (Dataset | str | Path): The reference dataset or its file path.
        pred_dataset (Dataset | str | Path): The predicted dataset or its file path.
        problem_def (ProblemDefinition | str | Path): The common problem for the reference and predicted dataset
        scalar (str | int): The name of the scalar to study or its index.
        save_file_name (str, optional): Figure name when saving to PNG format. Defaults to "bissec_plots".
        verbose (bool, optional): Verbose mode. Defaults to False.

    Raises:
        KeyError: If the provided scalar name is not part of the dataset.
    """
    ### Transform path to Dataset object ###
    if isinstance(ref_dataset, (str, Path)):
        ref_dataset: Dataset = Dataset(ref_dataset)
    if isinstance(pred_dataset, (str, Path)):
        pred_dataset: Dataset = Dataset(pred_dataset)
    if isinstance(problem_def, (str, Path)):
        problem_def: ProblemDefinition = ProblemDefinition(problem_def)

    # Load the testing_set
    # testing_set = problem_def.get_split("test")

    print("Data preprocessing...") if verbose else None
    ref_out_scalars, pred_out_scalars, out_scalars_names = prepare_datasets(
        ref_dataset, pred_dataset, problem_def, verbose
    )

    ### Transform string to index ###
    if isinstance(scalar, str):
        if scalar in out_scalars_names:
            scalar: int = out_scalars_names.index(scalar)
        else:
            raise KeyError(
                f"The scalar name provided ({scalar}) is not part of '{out_scalars_names = }'"
            )

    # Matplotlib plotting options
    if is_dvipng_available(verbose):  # pragma: no cover
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"],
            }
        )
        mpl.style.use("seaborn-v0_8")
    else:  # pragma: no cover
        mpl.rcParams.update(mpl.rcParamsDefault)

    fontsize = 32
    labelsize = 32
    markersize = 24
    markeredgewidth = 1

    #### Bisect graph plot ####
    print("Bisect graph construction...") if verbose else None
    label = r"$\mathrm{Predictions~vs~Targets~for~" + out_scalars_names[scalar] + "}$"
    fig, ax = plt.subplots(figsize=(2 * 6, 2 * 5.5))

    ### Matplotlib instructions ###
    y_true_dataset = np.array(
        ref_out_scalars[out_scalars_names[scalar]]
    )  # [testing_set]
    y_pred_dataset = np.array(
        pred_out_scalars[out_scalars_names[scalar]]
    )  # [testing_set]

    m, M = np.min(y_true_dataset), np.max(y_true_dataset)
    ax.plot(np.array([m, M]), np.array([m, M]), color="k")

    ax.plot(
        y_true_dataset,
        y_pred_dataset,
        linestyle="",
        color="b",
        markerfacecolor="r",
        markeredgecolor="b",
        markeredgewidth=markeredgewidth,
        marker=".",
        markersize=markersize,
    )

    ax.tick_params(labelsize=labelsize)
    ax.set_title(label, fontsize=fontsize)

    ax.set_ylabel(r"$\mathrm{Predictions}$", fontsize=fontsize)
    ax.set_xlabel(r"$\mathrm{Targets}$", fontsize=fontsize)

    plt.tight_layout()

    print("Bisect graph saving...") if verbose else None
    fig.savefig(f"{save_file_name}.png", dpi=300, format="png", bbox_inches="tight")
    print("...Bisect plot done") if verbose else None
