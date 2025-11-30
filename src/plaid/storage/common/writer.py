import io
import logging
import pickle
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Union

import yaml
from huggingface_hub import HfApi

from plaid import ProblemDefinition

logger = logging.getLogger(__name__)

# ------------------------------------------------------
# Write to disk
# ------------------------------------------------------


def _check_folder(output_folder: Path, overwrite: bool) -> None:
    if output_folder.is_dir():
        if overwrite:
            shutil.rmtree(output_folder)
            logger.warning(f"Existing {output_folder} directory has been reset.")
        elif any(output_folder.iterdir()):
            raise ValueError(
                f"directory {output_folder} already exists and is not empty. Set `overwrite` to True if needed."
            )


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


def save_problem_definitions_to_disk(
    path: Union[str, Path],
    pb_defs: Union[ProblemDefinition, Iterable[ProblemDefinition]],
) -> None:
    """Save a ProblemDefinition and its split information to disk.

    Args:
        path (Union[str, Path]): The root directory path for saving.
        name (str): The name of the problem_definition to store in the disk directory.
        pb_def (ProblemDefinition): The problem definition to save.
    """
    if isinstance(pb_defs, ProblemDefinition):
        pb_defs = [pb_defs]

    if not isinstance(pb_defs, Iterable):
        raise TypeError(
            f"pb_defs must be a ProblemDefinition or an iterable, got {type(pb_defs)}"
        )

    target_dir = Path(path) / "problem_definitions"
    target_dir.mkdir(parents=True, exist_ok=True)

    for pb_def in pb_defs:
        name = pb_def.get_name() or "default"
        pb_def.save_to_file(target_dir / name)


def save_metadata_to_disk(
    path: Union[str, Path],
    flat_cst: dict[str, Any],
    variable_schema: dict[str, Any],
    constant_schema: dict[str, Any],
    cgns_types: dict[str, Any]
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

    with open(Path(path) / "variable_schema.yaml", "w", encoding="utf-8") as f:
        yaml.dump(variable_schema, f, sort_keys=False)

    with open(Path(path) / "constant_schema.yaml", "w", encoding="utf-8") as f:
        yaml.dump(constant_schema, f, sort_keys=False)

    with open(Path(path) / "cgns_types.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cgns_types, f, sort_keys=False)


# ------------------------------------------------------
# Push to hub
# ------------------------------------------------------


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


def push_problem_definitions_to_hub(
    repo_id: str, pb_defs: Union[ProblemDefinition, Iterable[ProblemDefinition]]
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload a ProblemDefinition and its split information to the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.
        name (str): The name of the problem_definition to store in the repo.
        pb_def (ProblemDefinition): The problem definition to upload.
    """
    if isinstance(pb_defs, ProblemDefinition):
        pb_defs = [pb_defs]

    if not isinstance(pb_defs, Iterable):
        raise TypeError(
            f"pb_defs must be a ProblemDefinition or an iterable, got {type(pb_defs)}"
        )

    api = HfApi()

    for pb_def in pb_defs:
        name = pb_def.get_name() or "default"
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


def push_metadata_to_hub(
    repo_id: str,
    flat_cst: dict[str, Any],
    variable_schema: dict[str, Any],
    constant_schema: dict[str, Any],
    cgns_types: dict[str, Any]
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
    yaml_str = yaml.dump(variable_schema, sort_keys=False)
    yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo="variable_schema.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload variable_schema.yaml",
    )

    # var_features_types
    yaml_str = yaml.dump(constant_schema, sort_keys=False)
    yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo="constant_schema.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload constant_schema.yaml",
    )

    # cgns_types
    yaml_str = yaml.dump(cgns_types, sort_keys=False)
    yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    api.upload_file(
        path_or_fileobj=yaml_buffer,
        path_in_repo="cgns_types.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload cgns_types.yaml",
    )