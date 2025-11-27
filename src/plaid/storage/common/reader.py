
import pickle
from pathlib import Path
from typing import Any, Union

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from plaid import ProblemDefinition


#------------------------------------------------------
# Load from disk
#------------------------------------------------------

def load_infos_from_disk(path: Union[str, Path]) -> dict[str, dict[str, str]]:
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


def load_problem_definition_from_disk(
    path: Union[str, Path], name: Union[str, Path]
) -> ProblemDefinition:
    """Load a ProblemDefinition and its split information from disk.

    Args:
        path (Union[str, Path]): The root directory path for loading.
        name (str): The name of the problem_definition stored in the disk directory.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    pb_def = ProblemDefinition()
    pb_def._load_from_file_(Path(path) / Path("problem_definitions") / Path(name))
    return pb_def


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

    cgns_types = {}
    for path in variable_schema.keys():
        cgns_types[path] = variable_schema[path]["cgns_type"]
    for path in constant_schema.keys():
        cgns_types[path] = constant_schema[path]["cgns_type"]

    return flat_cst, variable_schema, constant_schema, cgns_types


#------------------------------------------------------
# Load from from hub
#------------------------------------------------------

def download_repo(
    repo_id: str,
    local_dir: Union[str, Path],
)-> str:  # pragma: no cover (not tested in unit tests)

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir
    )


def load_infos_from_hub(
    repo_id: str,
) -> dict[str, dict[str, str]]:  # pragma: no cover (not tested in unit tests)
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


def load_problem_definition_from_hub(
    repo_id: str, name: str
) -> ProblemDefinition:  # pragma: no cover (not tested in unit tests)
    """Load a ProblemDefinition from the Hugging Face Hub.

    Downloads the problem infos YAML and split JSON files from the specified repository and location,
    then initializes a ProblemDefinition object with this information.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        name (str): The name of the problem_definition stored in the repo.

    Returns:
        ProblemDefinition: The loaded problem definition.
    """
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    # Download problem_infos.yaml
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"problem_definitions/{name}",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    prob_def = ProblemDefinition()
    prob_def._initialize_from_problem_infos_dict(yaml_data)

    return prob_def


def load_metadata_from_hub(
    repo_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:  # pragma: no cover (not tested in unit tests)
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

    cgns_types = {}
    for path in variable_schema.keys():
        cgns_types[path] = variable_schema[path]["cgns_type"]
    for path in constant_schema.keys():
        cgns_types[path] = constant_schema[path]["cgns_type"]

    return flat_cst, variable_schema, constant_schema, cgns_types


