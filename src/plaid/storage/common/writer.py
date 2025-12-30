# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Common storage writer utilities.

This module provides common utilities for writing dataset metadata, problem definitions,
and other auxiliary files to disk or uploading them to Hugging Face Hub. It handles
serialization of infos, problem definitions, and dataset tree structures.
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml
from huggingface_hub import HfApi

from plaid import ProblemDefinition

logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Write to disk
# ------------------------------------------------------


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
    pb_defs: Union[dict[str, ProblemDefinition], ProblemDefinition],
) -> None:
    """Save ProblemDefinitions to disk.

    Args:
        path (Union[str, Path]): The directory path for saving.
        pb_defs (Union[dict[str, ProblemDefinition], ProblemDefinition]): The problem definitions to save.
    """
    if isinstance(pb_defs, ProblemDefinition):
        pb_defs = {pb_defs.get_name(): pb_defs}

    target_dir = Path(path) / "problem_definitions"
    target_dir.mkdir(parents=True, exist_ok=True)

    for name, pb_def in pb_defs.items():
        if name is None:
            raise ValueError(
                "At least one of the provided pb_defs has no initialized name."
            )
        pb_def.save_to_file(target_dir / name)


# def save_constants_to_disk(path, constant_schema, flat_cst):

#     for split in flat_cst.keys():

#         layout = {}
#         offset = 0

#         cst_path = path / "constants" / split
#         cst_path.mkdir(parents=True, exist_ok=True)

#         with open(cst_path / "data.mmap", "wb") as f:
#             for key, spec in constant_schema[split].items():

#                 dtype = spec["dtype"]

#                 if dtype is None:
#                     layout[key] = None
#                     continue

#                 value = flat_cst[split][key]
#                 arr = value

#                 assert arr.ndim == spec["ndim"]

#                 if dtype == "string":
#                     arr_bytes = np.asarray(arr, dtype="<U1").astype("|S1")

#                     f.write(arr_bytes.tobytes(order="C"))
#                     nbytes = arr_bytes.nbytes

#                 else:
#                     f.write(arr.tobytes(order="C"))
#                     nbytes = arr.nbytes

#                 layout[key] = {
#                     "offset": offset,
#                     "shape": list(arr.shape),
#                 }

#                 offset += nbytes

#         json.dump(layout, open(cst_path / "layout.json", "w"), indent=2)

#         with open(cst_path / "constant_schema.yaml", "w", encoding="utf-8") as f:
#             yaml.dump(constant_schema[split], f, sort_keys=False)


def save_constants_to_disk(path, constant_schema, flat_cst):
    """Write constant features to disk under <path>/constants/.

    For each split in flat_cst this creates a directory:
      <path>/constants/<split>/
        - data.mmap            : concatenated raw bytes of all constants for that split
        - layout.json          : mapping constant_name -> {'offset': int, 'shape': [...] } or None
        - constant_schema.yaml : the provided schema for that split (dtype and ndim)

    Behavior:
      - Numeric constants are written as their C-order bytes.
      - String constants support two cases:
          * CGNS string scalar: a 1-element array of Python str -> written as ASCII bytes, shape recorded as [len].
          * CGNS char array: multi-char arrays -> converted to fixed-width bytes and written.
      - If a schema entry's dtype is None, the layout entry is set to None and no bytes are written.

    Args:
        path (str | Path): Root dataset directory where "constants" will be created.
        constant_schema (dict): Mapping split -> {constant_name: {'dtype': str | None, 'ndim': int, ...}}.
        flat_cst (dict): Mapping split -> {constant_name: numpy array | None} containing values to save.

    Returns:
        None

    Raises:
        AssertionError: if a numeric array does not match the expected ndim.
        OSError / IOError: on file system write errors.
    """
    for split in flat_cst.keys():
        layout = {}
        offset = 0

        cst_path = path / "constants" / split
        cst_path.mkdir(parents=True, exist_ok=True)

        with open(cst_path / "data.mmap", "wb") as f:
            for key, spec in constant_schema[split].items():
                dtype = spec["dtype"]

                if dtype is None:
                    layout[key] = None
                    continue

                value = flat_cst[split][key]

                # -----------------
                # STRING CASE
                # -----------------
                if dtype == "string":
                    arr = np.asarray(value)

                    # ---- CASE 1: CGNS string scalar ----
                    if arr.ndim == 1 and arr.size == 1:
                        s = arr[0]
                        assert isinstance(s, str)

                        raw = s.encode("ascii", "strict")
                        f.write(raw)

                        shape = [len(raw)]
                        nbytes = len(raw)

                        layout[key] = {
                            "offset": offset,
                            "shape": shape,
                            "dtype": "|S1",
                            "order": "C",
                        }

                    # ---- CASE 2: CGNS char array ----
                    else:  # pragma: no cover
                        arr = arr.astype("<U1")
                        arr_bytes = arr.astype("|S1")

                        f.write(arr_bytes.tobytes(order="C"))

                        shape = list(arr.shape)
                        nbytes = arr_bytes.nbytes

                        layout[key] = {
                            "offset": offset,
                            "shape": shape,
                            "dtype": "|S1",
                            "order": "C",
                        }

                # -----------------
                # NUMERIC CASE
                # -----------------
                else:
                    arr = np.asarray(value)
                    assert arr.ndim == spec["ndim"]

                    # FORCE contiguous + little-endian
                    arr = np.ascontiguousarray(arr)
                    arr = arr.astype(arr.dtype.newbyteorder("<"), copy=False)

                    f.write(arr.tobytes(order="C"))

                    shape = list(arr.shape)
                    nbytes = arr.nbytes

                    layout[key] = {
                        "offset": offset,
                        "shape": shape,
                        "dtype": arr.dtype.str,  # e.g. '<f8'
                        "order": "C",
                    }

                offset += nbytes

        json.dump(layout, open(cst_path / "layout.json", "w"), indent=2)

        with open(cst_path / "constant_schema.yaml", "w", encoding="utf-8") as f:
            yaml.dump(constant_schema[split], f, sort_keys=False)


def save_metadata_to_disk(
    path: Union[str, Path],
    flat_cst: dict[str, Any],
    variable_schema: dict[str, Any],
    constant_schema: dict[str, Any],
    cgns_types: dict[str, Any],
) -> None:
    """Save the structure of a dataset tree to disk.

    This function writes the constant part of the tree and its key mappings to files
    in the specified directory. The constant part is serialized as a pickle file,
    while the key mappings are saved in YAML format.

    Args:
        path (Union[str, Path]): Directory path where the tree structure files will be saved.
        flat_cst (dict): Dictionary containing the constant part of the tree.
        variable_schema (dict): Dictionary containing the variable schema.
        constant_schema (dict): Dictionary containing the constant schema.
        cgns_types (dict): Dictionary containing CGNS types.

    Returns:
        None
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    save_constants_to_disk(path, constant_schema, flat_cst)

    with open(path / "variable_schema.yaml", "w", encoding="utf-8") as f:
        yaml.dump(variable_schema, f, sort_keys=False)

    with open(path / "cgns_types.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cgns_types, f, sort_keys=False)


# ------------------------------------------------------
# Push to hub
# ------------------------------------------------------


def push_infos_to_hub(
    repo_id: str, infos: dict[str, dict[str, str]]
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload dataset infos.yaml to a Hugging Face dataset repository.

    Serializes the provided `infos` mapping to YAML and uploads it as `infos.yaml`
    to the target `repo_id` using the HfApi.

    Args:
        repo_id (str): Hugging Face dataset repository identifier (e.g. "user/repo").
        infos (dict[str, dict[str, str]]): Dataset infos mapping to serialize and upload.

    Raises:
        ValueError: If `infos` is empty.
        OSError / IOError: If the upload fails due to I/O errors or network problems.

    Notes:
        - The function uses HfApi.upload_file and constructs the file contents in-memory.
        - Not covered by unit tests (pragma: no cover).
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


def push_local_problem_definitions_to_hub(
    repo_id: str,
    path: Union[Path, str],
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload local ProblemDefinitions to a Hugging Face dataset repository.

    This function uploads the entire local ``problem_definitions/`` directory
    located under ``path`` to the target Hugging Face dataset repository using
    ``HfApi.upload_folder``.

    Expected local layout:
        <path>/
            problem_definitions/
                <name_1>
                <name_2>
                ...

    Each problem definition is assumed to already be serialized on disk
    (e.g. via ``ProblemDefinition.save_to_file``). The function performs a
    directory-level upload and does not inspect, validate, or re-serialize
    individual problem definitions.

    Args:
        repo_id (str):
            Hugging Face dataset repository identifier
            (e.g. ``"username/dataset_name"``).

        path (Union[Path, str]):
            Root dataset directory containing the ``problem_definitions/`` folder.

    Notes:
        - Upload is atomic at the folder level.
        - Existing files in ``problem_definitions/`` on the Hub may be overwritten.
        - Uses ``repo_type="dataset"``.
        - Not covered by unit tests (``pragma: no cover``).

    Raises:
        OSError / IOError:
            If the local folder does not exist or an upload error occurs.
    """
    path = Path(path)

    api = HfApi()

    api.upload_folder(
        folder_path=path / Path("problem_definitions"),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="problem_definitions",
        commit_message="Upload problem_definitions",
    )

    # pb_defs = load_problem_definitions_from_disk(path)
    # if isinstance(pb_defs, ProblemDefinition):
    #     pb_defs = [pb_defs]

    # if not isinstance(pb_defs, Iterable):
    #     raise TypeError(
    #         f"pb_defs must be a ProblemDefinition or an iterable, got {type(pb_defs)}"
    #     )

    # api = HfApi()

    # for pb_def in pb_defs:
    #     name = pb_def.get_name() or "default"
    #     data = pb_def._generate_problem_infos_dict()
    #     for k, v in list(data.items()):
    #         if not v:
    #             data.pop(k)
    #     if data is not None:
    #         yaml_str = yaml.dump(data)
    #         yaml_buffer = io.BytesIO(yaml_str.encode("utf-8"))

    #     if not name.endswith(".yaml"):
    #         name = f"{name}.yaml"

    #     api.upload_file(
    #         path_or_fileobj=yaml_buffer,
    #         path_in_repo=f"problem_definitions/{name}",
    #         repo_id=repo_id,
    #         repo_type="dataset",
    #         commit_message=f"Upload problem_definitions/{name}",
    #     )


def push_local_metadata_to_hub(
    repo_id: str,
    path: Union[Path, str],
) -> None:  # pragma: no cover (not tested in unit tests)
    """Upload locally stored dataset metadata to a Hugging Face dataset repository.

    This function uploads the structural metadata of a PLAID dataset from disk
    to a Hugging Face Hub *dataset* repository. The upload consists of:

      1. The ``constants/`` directory, containing:
         - ``data.mmap`` files with concatenated constant values,
         - ``layout.json`` files describing byte offsets and shapes,
         - ``constant_schema.yaml`` files describing constant dtypes and dimensions,
         organized per dataset split.

      2. ``variable_schema.yaml``, describing the schema of variable (sample-dependent)
         features.

      3. ``cgns_types.yaml``, describing CGNS node types associated with dataset paths.

    All metadata files are assumed to have been previously generated on disk
    (e.g. via ``save_metadata_to_disk``). This function performs no validation,
    transformation, or serialization; it strictly uploads existing files.

    Expected local layout:
        <path>/
            constants/
                <split>/
                    data.mmap
                    layout.json
                    constant_schema.yaml
            variable_schema.yaml
            cgns_types.yaml

    Args:
        repo_id (str):
            Hugging Face dataset repository identifier
            (e.g. ``"username/dataset_name"``).

        path (Union[Path, str]):
            Root dataset directory containing the metadata files.

    Notes:
        - Uploads use ``repo_type="dataset"``.
        - Folder uploads may overwrite existing files on the Hub.
        - The operation is atomic per uploaded artifact
          (``constants/`` folder, individual YAML files).
        - Not covered by unit tests (``pragma: no cover``).

    Raises:
        OSError / IOError:
            If required local files are missing or if an upload error occurs.
    """
    api = HfApi()

    path = Path(path)

    # constant part of the tree
    api.upload_folder(
        folder_path=path / "constants",
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="constants",
        commit_message="Upload constants (memmap + layout)",
    )

    # variable_schema
    api.upload_file(
        path_or_fileobj=path / "variable_schema.yaml",
        path_in_repo="variable_schema.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload variable_schema.yaml",
    )

    # cgns_types
    api.upload_file(
        path_or_fileobj=path / "cgns_types.yaml",
        path_in_repo="cgns_types.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload cgns_types.yaml",
    )
