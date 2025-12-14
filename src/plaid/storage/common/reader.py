import pickle
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from plaid import ProblemDefinition

# ------------------------------------------------------
# Load from disk
# ------------------------------------------------------


def load_infos_from_disk(path: Union[str, Path]) -> dict[str, Any]:
    """Load dataset information from a YAML file stored on disk.

    Args:
        path (Union[str, Path]): Directory path containing the `infos.yaml` file.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing dataset infos.
    """
    infos_fname = Path(path) / "infos.yaml"
    with infos_fname.open("r") as file:
        infos = yaml.safe_load(file)
    return infos


def load_problem_definitions_from_disk(
    path: Union[str, Path],
) -> Optional[list[ProblemDefinition]]:
    """Load a ProblemDefinition and its split information from disk.

    Args:
        path (Union[str, Path]): The root directory path for loading.
        name (str): The name of the problem_definition stored in the disk directory.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    pb_def_dir = Path(path) / Path("problem_definitions")

    if pb_def_dir.is_dir():
        pb_defs = []
        for p in pb_def_dir.iterdir():
            if p.is_file():
                pb_def = ProblemDefinition()
                pb_def._load_from_file_(pb_def_dir / Path(p.name))
                pb_defs.append(pb_def)
        return pb_defs
    else:
        return None  # pragma: no cover


def load_metadata_from_disk(
    path: Union[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a tree structure for a dataset from disk.

    This function loads two components from the specified directory:
    1. `tree_constant_part.pkl`: a pickled dictionary containing the constant parts of the tree.
    2. `key_mappings.yaml`: a YAML file containing key mappings and metadata.

    Args:
        path (Union[str, Path]): Directory path containing the `tree_constant_part.pkl`
            and `key_mappings.yaml` files.

    Returns:
        tuple[dict, dict]: A tuple containing:
            - `flat_cst` (dict): Dictionary of constant tree values.
            - `key_mappings` (dict): Dictionary of key mappings and metadata.
    """
    with open(Path(path) / "tree_constant_part.pkl", "rb") as f:
        flat_cst = pickle.load(f)

    with open(Path(path) / Path("variable_schema.yaml"), "r", encoding="utf-8") as f:
        variable_schema = yaml.safe_load(f)

    with open(Path(path) / Path("constant_schema.yaml"), "r", encoding="utf-8") as f:
        constant_schema = yaml.safe_load(f)

    with open(Path(path) / Path("cgns_types.yaml"), "r", encoding="utf-8") as f:
        cgns_types = yaml.safe_load(f)

    return flat_cst, variable_schema, constant_schema, cgns_types


# ------------------------------------------------------
# Load from from hub
# ------------------------------------------------------


def load_infos_from_hub(
    repo_id: str,
) -> dict[str, Any]:  # pragma: no cover
    """Load dataset infos from the Hugging Face Hub.

    Downloads the infos.yaml file from the specified repository and parses it as a dictionary.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        dict[str, dict[str, str]]: Dictionary containing dataset infos.
    """
    # Download infos.yaml
    yaml_path = hf_hub_download(
        repo_id=repo_id, filename="infos.yaml", repo_type="dataset"
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        infos = yaml.safe_load(f)

    return infos


def load_problem_definitions_from_hub(
    repo_id: str,
) -> Optional[list[ProblemDefinition]]:  # pragma: no cover
    """Load a ProblemDefinition from the Hugging Face Hub.

    Downloads the problem infos YAML and split JSON files from the specified repository and location,
    then initializes a ProblemDefinition object with this information.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    with tempfile.TemporaryDirectory(prefix="pb_def_") as temp_folder:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["problem_definitions/"],
            local_dir=temp_folder,
        )
        pb_defs = load_problem_definitions_from_disk(temp_folder)
    return pb_defs


def load_metadata_from_hub(
    repo_id: str,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:  # pragma: no cover
    """Load the tree structure metadata of a PLAID dataset from the Hugging Face Hub.

    This function retrieves two artifacts previously uploaded alongside a dataset:
      - **tree_constant_part.pkl**: a pickled dictionary of constant feature values
        (features that are identical across all samples).
      - **key_mappings.yaml**: a YAML file containing metadata about the dataset
        feature structure, including variable features, constant features, and CGNS types.

    Args:
        repo_id (str):
            The repository ID on the Hugging Face Hub
            (e.g., `"username/dataset_name"`).

    Returns:
        tuple[dict, dict]:
            - **flat_cst (dict)**: constant features dictionary (path → value).
            - **key_mappings (dict)**: metadata dictionary containing keys such as:
                - `"variable_features"`: list of paths for non-constant features.
                - `"constant_features"`: list of paths for constant features.
                - `"cgns_types"`: mapping from paths to CGNS types.
    """
    # constant part of the tree
    flat_cst_path = hf_hub_download(
        repo_id=repo_id,
        filename="tree_constant_part.pkl",
        repo_type="dataset",
    )

    with open(flat_cst_path, "rb") as f:
        flat_cst = pickle.load(f)

    # variable_schema
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename="variable_schema.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        variable_schema = yaml.safe_load(f)

    # constant_schema
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename="constant_schema.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        constant_schema = yaml.safe_load(f)

    # cgns_types
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename="cgns_types.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        cgns_types = yaml.safe_load(f)

    return flat_cst, variable_schema, constant_schema, cgns_types
