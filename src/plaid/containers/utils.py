"""Utility functions for PLAID containers."""
# %% Imports

from pathlib import Path
from typing import Any, Optional, Union

import CGNS.PAT.cgnsutils as CGU
import numpy as np

from ..constants import (
    CGNS_FIELD_LOCATIONS,
)

from ..types.common import ArrayDType
from ..utils.base import safe_len

path_to_location = {f"{loc}Fields": loc for loc in CGNS_FIELD_LOCATIONS}
retrocompatibility = {
    "PointData": "Vertex",
    "CellData": "CellCenter",
    "SurfaceData": "FaceCenter",
}
path_to_location.update(retrocompatibility)


def _check_names(names: Union[str, list[Optional[str]], None]):
    """Check that names do not contain invalid character ``/``.

    Args:
        names (Union[str, list[Optional[str]], None]): The names to check.

    Raises:
        ValueError: If any name contains the invalid character ``/``.
    """
    if names is None:
        names = [None]
    if isinstance(names, str):
        names = [names]
    for name in names:
        if (name is not None) and ("/" in name):
            raise ValueError(
                f"feature_names containing `/` are not allowed, but {name=}, you should first replace any occurrence of `/` with something else, for example: `name.replace('/','__')`"
            )
        if (name is not None) and (len(name) > 32):
            raise ValueError(
                f"CGNS names must be shorter than or equal to 32 characters, but got {name=} with length {len(name)}"
            )


def _read_index(pyTree: list, dim: list[int]):
    """Read Index Array or Index Range from CGNS.

    Args:
        pyTree (list): CGNS node which has a child Index to read
        dim (list): dimensions of the coordinates

    Returns:
        indices
    """
    a = _read_index_array(pyTree)
    b = _read_index_range(pyTree, dim)
    return np.hstack((a, b))


def _read_index_array(pyTree: list):
    """Read Index Array from CGNS.

    Args:
        pyTree (list): CGNS node which has a child of type IndexArray_t to read

    Returns:
        indices
    """
    indexArrayPaths = CGU.getPathsByTypeSet(pyTree, ["IndexArray_t"])
    res = []
    for indexArrayPath in indexArrayPaths:
        data = CGU.getNodeByPath(pyTree, indexArrayPath)
        if data[1] is None:  # pragma: no cover
            continue
        else:
            res.extend(data[1].ravel())
    return np.array(res, dtype=int).ravel()


def _read_index_range(pyTree: list, dim: list[int]):
    """Read Index Range from CGNS.

    Args:
        pyTree (list): CGNS node which has a child of type IndexRange_t to read
        dim (list[str]): dimensions of the coordinates

    Returns:
        indices
    """
    indexRangePaths = CGU.getPathsByTypeSet(pyTree, ["IndexRange_t"])
    res = []

    for indexRangePath in indexRangePaths:  # Is it possible there are several ?
        indexRange = CGU.getValueByPath(pyTree, indexRangePath)

        if indexRange.shape == (3, 2):  # 3D  # pragma: no cover
            for k in range(indexRange[:, 0][2], indexRange[:, 1][2] + 1):
                for j in range(indexRange[:, 0][1], indexRange[:, 1][1] + 1):
                    global_id = (
                        np.arange(indexRange[:, 0][0], indexRange[:, 1][0] + 1)
                        + dim[0] * (j - 1)
                        + dim[0] * dim[1] * (k - 1)
                    )
                    res.extend(global_id)

        elif indexRange.shape == (2, 2):  # 2D  # pragma: no cover
            for j in range(indexRange[:, 0][1], indexRange[:, 1][1]):
                for i in range(indexRange[:, 0][0], indexRange[:, 1][0]):
                    global_id = i + dim[0] * (j - 1)
                    res.append(global_id)
        else:
            begin = indexRange[0]
            end = indexRange[1]
            res.extend(np.arange(begin, end + 1).ravel())

    return np.array(res, dtype=int).ravel()


def get_sample_ids(savedir: Union[str, Path]) -> list[int]:
    """Return list of sample ids from a dataset on disk.

    Args:
        savedir (Union[str,Path]): The path to the directory where sample files are stored.

    Returns:
        list[int]: List of sample ids.
    """
    savedir = Path(savedir)
    return sorted(
        [int(d.stem.split("_")[-1]) for d in (savedir).glob("sample_*") if d.is_dir()]
    )


def get_number_of_samples(savedir: Union[str, Path]) -> int:
    """Return number of samples in a dataset on disk.

    Args:
        savedir (Union[str,Path]): The path to the directory where sample files are stored.

    Returns:
        int: number of samples.
    """
    return len(get_sample_ids(savedir))


def get_feature_details_from_path(path: str) -> dict[str, str]:
    """Retrieve semantic details from a CGNS-style path."""
    split_path = path.split("/")
    feat: dict[str, str] = {}

    # ----------------------
    # Global
    # ----------------------
    if split_path[0] == "Global" or split_path[0] == "Global_times":
        feat["type"] = "global"

        if len(split_path) == 1:
            feat["sub_type"] = "root"

        elif split_path[1] == "Time":
            feat["sub_type"] = "time"
            if len(split_path) == 3:
                feat["name"] = split_path[2]

        else:
            feat["sub_type"] = "scalar"
            feat["name"] = split_path[-1]

        return feat

    # ----------------------
    # CGNS library version
    # ----------------------
    if path == "CGNSLibraryVersion":
        return {
            "type": "cgns",
            "sub_type": "library_version",
        }

    # ----------------------
    # Base / Zone
    # ----------------------
    feat["base"] = split_path[0]
    assert feat["base"].startswith("Base_"), "path not recognized"

    if len(split_path) == 1:
        feat["type"] = "base"
        return feat

    feat["zone"] = split_path[1]

    if len(split_path) == 2:
        feat["type"] = "zone"
        return feat

    node = split_path[2]

    # ----------------------
    # Grid coordinates
    # ----------------------
    if node == "GridCoordinates" and len(split_path) >= 3 and len(split_path) <= 4:
        feat["type"] = "coordinate"
        feat["sub_type"] = "node"
        if len(split_path) == 4:
            feat["name"] = split_path[3]
        return feat

    # ----------------------
    # Elements
    # ----------------------
    if node.startswith("Elements_") and len(split_path) == 4:
        feat["type"] = "elements"
        feat["element_type"] = node[len("Elements_") :]

        leaf = split_path[3]
        if leaf == "ElementConnectivity":
            feat["sub_type"] = "connectivity"
        elif leaf == "ElementRange":
            feat["sub_type"] = "range"

        return feat

    # ----------------------
    # Boundary conditions
    # ----------------------
    if node == "ZoneBC" and len(split_path) >= 4:
        feat["type"] = "boundary_condition"
        feat["name"] = split_path[3]

        if len(split_path) == 4:
            feat["sub_type"] = "bc"
        elif len(split_path) == 5:
            feat["sub_type"] = split_path[4]  # PointList or GridLocation

        return feat

    # ----------------------
    # Fields (generic location)
    # ----------------------
    if node in path_to_location:
        feat["type"] = "field"
        feat["location"] = path_to_location[node]

        if len(split_path) == 4:
            feat["name"] = split_path[3]

        return feat

    # ----------------------
    # Fallback
    # ----------------------
    feat["type"] = "other"
    feat["path"] = path
    return feat


