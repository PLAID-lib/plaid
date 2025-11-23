from typing import Any, Union
from pathlib import Path
import yaml
import pickle
import io

from plaid import ProblemDefinition

from huggingface_hub import HfApi

#------------------------------------------------------
# Write to disk
#------------------------------------------------------

def save_infos_to_disk(
    path: Union[str, Path], infos: dict[str, dict[str, str]]
) -> None:
    """Save dataset infos as a YAML file to disk.

    Args:
        path (Union[str, Path]): The directory path where the infos file will be saved.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset infos.
    """
    infos_fname = Path(path) / "infos.yaml"
    infos_fname.parent.mkdir(parents=True, exist_ok=True)
    with open(infos_fname, "w") as file:
        yaml.dump(infos, file, default_flow_style=False, sort_keys=False)


def save_problem_definition_to_disk(
    path: Union[str, Path], name: Union[str, Path], pb_def: ProblemDefinition
) -> None:
    """Save a ProblemDefinition and its split information to disk.

    Args:
        path (Union[str, Path]): The root directory path for saving.
        name (str): The name of the problem_definition to store in the disk directory.
        pb_def (ProblemDefinition): The problem definition to save.
    """
    pb_def.save_to_file(Path(path) / Path("problem_definitions") / Path(name))


def save_tree_struct_to_disk(
    path: Union[str, Path],
    flat_cst: dict[str, Any],
    key_mappings: dict[str, Any],
) -> None:
    """Save the structure of a dataset tree to disk.

    This function writes the constant part of the tree and its key mappings to files
    in the specified directory. The constant part is serialized as a pickle file,
    while the key mappings are saved in YAML format.

    Args:
        path (Union[str, Path]): Directory path where the tree structure files will be saved.
        flat_cst (dict): Dictionary containing the constant part of the tree.
        key_mappings (dict): Dictionary containing key mappings for the tree structure.

    Returns:
        None
    """
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(Path(path) / "tree_constant_part.pkl", "wb") as f:
        pickle.dump(flat_cst, f)

    with open(Path(path) / "key_mappings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(key_mappings, f, sort_keys=False)


#------------------------------------------------------
# Push to hub
#------------------------------------------------------

def push_infos_to_hub(
    repo_id: str, infos: dict[str, dict[str, str]]
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload dataset infos to the Hugging Face Hub.

    Serializes the infos dictionary to YAML and uploads it to the specified repository as infos.yaml.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        infos (dict[str, dict[str, str]]): Dictionary containing dataset infos to upload.

    Raises:
        ValueError: If the infos dictionary is empty.
    """
    if len(infos) > 0:
        api = HfApi()
        yaml_str = yaml.dump(infos)
        yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))
        api.upload_file(
            path_or_fileobj=yaml_buffer,
            path_in_repo="infos.yaml",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload infos.yaml",
        )
    else:
        raise ValueError("'infos' must not be empty")


def push_problem_definition_to_hub(
    repo_id: str, name: str, pb_def: ProblemDefinition
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload a ProblemDefinition and its split information to the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        name (str): The name of the problem_definition to store in the repo.
        pb_def (ProblemDefinition): The problem definition to upload.
    """
    api = HfApi()
    data = pb_def._generate_problem_infos_dict()
    for k, v in list(data.items()):
        if not v:
            data.pop(k)
    if data is not None:
        yaml_str = yaml.dump(data)
        yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    if not name.endswith(".yaml"):
        name = f"{name}.yaml"

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo=f"problem_definitions/{name}",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload problem_definitions/{name}",
    )


def push_tree_struct_to_hub(
    repo_id: str,
    flat_cst: dict[str, Any],
    key_mappings: dict[str, Any],
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload a dataset's tree structure to a Hugging Face dataset repository.

    This function pushes two components of a dataset tree structure to the specified
    Hugging Face Hub repository:

    1. `flat_cst`: the constant parts of the dataset tree, serialized as a pickle file
       (`tree_constant_part.pkl`).
    2. `key_mappings`: the dictionary of key mappings and metadata for the dataset tree,
       serialized as a YAML file (`key_mappings.yaml`).

    Both files are uploaded using the Hugging Face `HfApi().upload_file` method.

    Args:
        repo_id (str): The Hugging Face dataset repository ID where files will be uploaded.
        flat_cst (dict[str, Any]): Dictionary containing constant values in the dataset tree.
        key_mappings (dict[str, Any]): Dictionary containing key mappings and additional metadata.

    Returns:
        None

    Note:
        - Each upload includes a commit message indicating the filename.
        - This function is not covered by unit tests (`pragma: no cover`).
    """
    api = HfApi()

    # constant part of the tree
    api.upload_file(
        path_or_fileobj=io.BytesIO(pickle.dumps(flat_cst)),
        path_in_repo="tree_constant_part.pkl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload tree_constant_part.pkl",
    )

    # key mappings
    yaml_str = yaml.dump(key_mappings, sort_keys=False)
    yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo="key_mappings.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload key_mappings.yaml",
    )