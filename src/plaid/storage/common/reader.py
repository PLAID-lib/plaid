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

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
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
) -> dict[str, ProblemDefinition]:
    """Load ProblemDefinitions from a local dataset directory.

    This function reads all serialized ``ProblemDefinition`` files located in the
    ``problem_definitions/`` subdirectory under ``path`` and reconstructs them
    into ``ProblemDefinition`` objects.

    Each file is loaded using ``ProblemDefinition._load_from_file_`` and inserted
    into a dictionary keyed by the problem definition name.

    Expected local layout:
        <path>/
            problem_definitions/
                <problem_name_1>
                <problem_name_2>
                ...

    Args:
        path (Union[str, Path]):
            Root dataset directory containing the ``problem_definitions/`` folder.

    Returns:
        dict[str, ProblemDefinition]:
            Mapping from problem definition names to loaded ``ProblemDefinition``
            objects.

    Raises:
        ValueError:
            If the ``problem_definitions/`` directory does not exist.
    """
    pb_def_dir = Path(path) / Path("problem_definitions")

    if pb_def_dir.is_dir():
        pb_defs = {}
        for p in pb_def_dir.iterdir():
            if p.is_file():
                pb_def = ProblemDefinition()
                pb_def._load_from_file_(pb_def_dir / Path(p.name))
                pb_defs[pb_def.get_name()] = pb_def
        return pb_defs
    else:
        raise ValueError("No problem definitions found on disk.")  # pragma: no cover


def load_constants_from_disk(path):
    """Load constant features stored under a dataset's "constants" directory.

    The function expects the following layout under <path>/constants/:
      - one folder per split (e.g. "train", "test", ...)
        each containing:
          * layout.json            : mapping constant_name -> {'offset': int, 'shape': [..]} or None
          * constant_schema.yaml   : YAML describing dtype for each constant (dtype string or "string")
          * data.mmap              : raw bytes memory-mapped file containing packed constant data

    Args:
        path (str | Path): Root dataset directory that contains the "constants" folder.

    Returns:
        tuple:
            flat_cst (dict[str, dict[str, Any]]): Mapping split -> {constant_name: numpy array | None}.
                - Numeric constants are returned as numpy arrays with the dtype and shape specified
                  in the schema.
                - String constants are returned as 1-element numpy arrays of Python str decoded using ASCII.
                - If layout entry for a key is None, the value is returned as None.
            constant_schema (dict[str, dict[str, Any]]): Mapping split -> loaded constant schema (from YAML).

    Raises:
        FileNotFoundError: If the expected "constants" directory or required files are missing.
    """
    cst_path = Path(path) / "constants"

    folders = [p for p in cst_path.iterdir() if p.is_dir()]
    splits = [folder.name for folder in folders]

    flat_cst = {}
    constant_schema = {}

    for folder, split in zip(folders, splits):
        with open(folder / "layout.json", "r", encoding="utf-8") as f:
            layout = json.load(f)

        with open(folder / "constant_schema.yaml", "r", encoding="utf-8") as f:
            constant_schema[split] = yaml.safe_load(f)

        mmap = np.memmap(
            folder / "data.mmap",
            mode="r",
            dtype=np.uint8,
        )

        flat_cst[split] = {}

        for key, spec in constant_schema[split].items():
            entry = layout[key]

            if entry is None:
                flat_cst[split][key] = None
                continue

            offset = entry["offset"]
            shape = entry["shape"]

            if spec["dtype"] == "string":
                nbytes = int(np.prod(shape))
                raw = mmap[offset : offset + nbytes].tobytes()
                flat_cst[split][key] = np.array([raw.decode("ascii", "strict")])

            else:
                dtype = np.dtype(spec["dtype"])
                nbytes = int(np.prod(shape)) * dtype.itemsize

                view = mmap[offset : offset + nbytes]
                flat_cst[split][key] = view.view(dtype).reshape(shape)

    return flat_cst, constant_schema


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
    path = Path(path)

    flat_cst, constant_schema = load_constants_from_disk(path)
    ########
    # with open(Path(path) / "tree_constant_part.pkl", "rb") as f:
    #     flat_cst = pickle.load(f)

    # with open(Path(path) / Path("constant_schema.yaml"), "r", encoding="utf-8") as f:
    #     constant_schema = yaml.safe_load(f)
    ########

    with open(path / "variable_schema.yaml", "r", encoding="utf-8") as f:
        variable_schema = yaml.safe_load(f)

    with open(path / "cgns_types.yaml", "r", encoding="utf-8") as f:
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
) -> Optional[dict[str, ProblemDefinition]]:  # pragma: no cover
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
    with tempfile.TemporaryDirectory(prefix="constants_") as temp_folder:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=["constants/"],
            local_dir=temp_folder,
        )
        flat_cst, constant_schema = load_constants_from_disk(temp_folder)

    # variable_schema
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename="variable_schema.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        variable_schema = yaml.safe_load(f)

    # cgns_types
    yaml_path = hf_hub_download(
        repo_id=repo_id,
        filename="cgns_types.yaml",
        repo_type="dataset",
    )
    with open(yaml_path, "r", encoding="utf-8") as f:
        cgns_types = yaml.safe_load(f)

    return flat_cst, variable_schema, constant_schema, cgns_types
