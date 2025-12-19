# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Common storage reader utilities.

This module provides common utilities for reading dataset metadata, problem definitions,
and other auxiliary files from disk or downloading them from Hugging Face Hub.
"""

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from huggingface_hub import hf_hub_download, snapshot_download

from plaid import ProblemDefinition

logger = logging.getLogger(__name__)

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
    """Load ProblemDefinitions from disk.

    Args:
        path (Union[str, Path]): The directory path for loading.

    Returns:
        Optional[list[ProblemDefinition]]: List of loaded problem definitions, or None if not found.
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
        logger.warning("No problem definitions found on disk.")
        return None  # pragma: no cover


def load_metadata_from_disk(
    path: Union[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load dataset metadata from disk.

    Args:
        path (Union[str, Path]): Directory path containing the metadata files.

    Returns:
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
            - flat_cst: constant features dictionary
            - variable_schema: variable schema dictionary
            - constant_schema: constant schema dictionary
            - cgns_types: CGNS types dictionary
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
    """Load ProblemDefinitions from Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        Optional[list[ProblemDefinition]]: List of loaded problem definitions, or None if not found.
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
    """Load dataset metadata from Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on the Hugging Face Hub.

    Returns:
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
            - flat_cst: constant features dictionary
            - variable_schema: variable schema dictionary
            - constant_schema: constant schema dictionary
            - cgns_types: CGNS types dictionary
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
