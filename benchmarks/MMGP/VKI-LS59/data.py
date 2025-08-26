import pickle
from pathlib import Path
from typing import Any, Literal, Optional

from datasets import load_dataset
from sklearn.model_selection import KFold

from plaid.bridges.huggingface_bridge import huggingface_dataset_to_plaid


def extract_split_data(
    split: Literal["train", "test", "traintest"],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Extract input and output dictionaries for all samples in a given split of a Plaid dataset.

    Args:
        dataset_dir (str): Path to the directory containing the 'huggingface' subfolder.
        split (Literal["train", "test", "traintest"]): Which split to extract;
            'train', 'test', or 'traintest' (concatenation of train and test).

    Returns:
        inputs (dict[str, list[Any]]): Keys are mesh/scalar names; values are lists of sample data.
        outputs (dict[str, list[Any]]): Keys are selected field/scalar names; values are lists of sample data.
    """
    # 1) Load the HuggingFace dataset from disk
    hf_dataset = load_dataset("PLAID-datasets/VKI-LS59", split="all_samples")

    # 2) Convert to Plaid and retrieve the problem definition
    plaid_dataset, problem_definition = huggingface_dataset_to_plaid(hf_dataset)

    # 3) Get sample indices for the requested split
    if split == "traintest":
        ids = problem_definition.get_split("train") + problem_definition.get_split(
            "test"
        )
    else:
        ids = problem_definition.get_split(split)

    # 4) Determine the base mesh name from the first sample
    sample0 = plaid_dataset[ids[0]]
    base_name = sample0.get_base_names()[0]

    # 5) Retrieve the names of all input scalars
    input_scalars = problem_definition.get_input_scalars_names()

    # Define exactly which outputs to extract
    FIELD_OUTPUTS = ["mach", "nut"]
    SCALAR_OUTPUTS = ["Q", "power", "Pr", "Tr", "eth_is", "angle_out"]

    inputs: dict[str, list[Any]] = {}
    outputs: dict[str, list[Any]] = {}

    # --- INPUTS ---
    # Mesh node coordinates
    inputs["nodes"] = [plaid_dataset[i].get_nodes(base_name=base_name) for i in ids]

    # Input scalar values
    for key in input_scalars:
        inputs[key] = [plaid_dataset[i].scalars.get(key) for i in ids]

    # --- OUTPUTS ---
    # Selected mesh field data
    for field_name in FIELD_OUTPUTS:
        outputs[field_name] = [
            plaid_dataset[i].get_field(field_name, base_name=base_name) for i in ids
        ]

    # Selected output scalar values
    for key in SCALAR_OUTPUTS:
        outputs[key] = [plaid_dataset[i].scalars.get(key) for i in ids]

    return inputs, outputs


def make_kfold_splits(
    inputs: dict[str, list[Any]],
    outputs: dict[str, list[Any]],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> list[
    tuple[
        dict[str, list[Any]],  # train inputs
        dict[str, list[Any]],  # train outputs
        dict[str, list[Any]],  # val inputs
        dict[str, list[Any]],  # val outputs
    ]
]:
    """Split inputs and outputs into K folds for cross‑validation.

    Args:
        inputs (dict[str, list[Any]]):
            Dictionary of input data where each key maps to a list of samples.
        outputs (dict[str, list[Any]]):
            Dictionary of output data where each key maps to a list of samples.
        n_splits (int, optional):
            Number of folds. Defaults to 5.
        shuffle (bool, optional):
            Whether to shuffle the data before splitting. Defaults to True.
        random_state (Optional[int], optional):
            Seed for reproducible shuffling. Defaults to None.

    Returns:
        list of tuples, one per fold, each containing:
            - train_inputs (dict)
            - train_outputs (dict)
            - val_inputs (dict)
            - val_outputs (dict)
    """
    # Number of samples inferred from the length of any input list
    n_samples = len(next(iter(inputs.values())))
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    splits: list[
        tuple[
            dict[str, list[Any]],
            dict[str, list[Any]],
            dict[str, list[Any]],
            dict[str, list[Any]],
        ]
    ] = []

    for train_idx, val_idx in kf.split(range(n_samples)):
        # Build train/val dicts by indexing into the lists
        train_inputs = {k: [v[i] for i in train_idx] for k, v in inputs.items()}
        val_inputs = {k: [v[i] for i in val_idx] for k, v in inputs.items()}
        train_outputs = {k: [v[i] for i in train_idx] for k, v in outputs.items()}
        val_outputs = {k: [v[i] for i in val_idx] for k, v in outputs.items()}

        splits.append((train_inputs, train_outputs, val_inputs, val_outputs))

    return splits


def dump_predictions(
    outputs_pred: dict[str, list[Any]], filename: str = "predictions.pkl"
) -> None:
    """Dump predicted outputs to a pickle file with the same structure as the reference.

    Args:
        outputs_pred (dict[str, list[Any]]):
            Predicted outputs containing keys
            'nut', 'mach', 'Q', 'power', 'Pr', 'Tr', 'eth_is', 'angle_out'.
        filename (str): Path to the output .pkl file.
    """
    FIELD_OUTPUTS = ["nut", "mach"]
    SCALAR_OUTPUTS = ["Q", "power", "Pr", "Tr", "eth_is", "angle_out"]

    # Build a list of dicts, one per sample
    n_samples = len(outputs_pred[FIELD_OUTPUTS[0]])
    predictions = []
    for i in range(n_samples):
        rec: dict[str, Any] = {}
        for fn in FIELD_OUTPUTS:
            rec[fn] = outputs_pred[fn][i]
        for sn in SCALAR_OUTPUTS:
            rec[sn] = outputs_pred[sn][i]
        predictions.append(rec)

    # Ensure output directory exists
    dump_path = Path(filename)
    dump_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to pickle
    with dump_path.open("wb") as f:
        pickle.dump(predictions, f)

    print(f"Predictions successfully dumped to '{dump_path}'")
