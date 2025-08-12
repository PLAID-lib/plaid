"""Implementation of the `Sample` container."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import copy
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import CGNS.MAP as CGM
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import numpy as np
from CGNS.PAT.cgnsutils import __CHILDREN__, __NAME__
from pydantic import BaseModel, model_serializer

from plaid.constants import (
    AUTHORIZED_FEATURE_INFOS,
    AUTHORIZED_FEATURE_TYPES,
    CGNS_ELEMENT_NAMES,
    CGNS_FIELD_LOCATIONS,
)
from plaid.containers.utils import get_feature_type_and_details_from
from plaid.types import (
    CGNSNode,
    CGNSTree,
    FeatureIdentifier,
    FeatureType,
    FieldType,
    LinkType,
    PathType,
    ScalarType,
    TimeSequenceType,
    TimeSeriesType,
)
from plaid.utils import cgns_helper as CGH
from plaid.utils.base import safe_len

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)

# %% Globals


# %% Classes


def _check_names(names: Union[str, list[str]]):
    """Check that names do not contain invalid character ``/``.

    Args:
        names (Union[str, list[str]]): The names to check.

    Raises:
        ValueError: If any name contains the invalid character ``/``.
    """
    if isinstance(names, str):
        names = [names]
    for name in names:
        if (name is not None) and ("/" in name):
            raise ValueError(
                f"feature_names containing `/` are not allowed, but {name=}, you should first replace any occurence of `/` with something else, for example: `name.replace('/','__')`"
            )


def read_index(pyTree: list, dim: list[int]):
    """Read Index Array or Index Range from CGNS.

    Args:
        pyTree (list): CGNS node which has a child Index to read
        dim (list): dimensions of the coordinates

    Returns:
        indices
    """
    a = read_index_array(pyTree)
    b = read_index_range(pyTree, dim)
    return np.hstack((a, b))


def read_index_array(pyTree: list):
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


def read_index_range(pyTree: list, dim: list[int]):
    """Read Index Range from CGNS.

    Args:
        pyTree (list): CGNS node which has a child of type IndexRange_t to read
        dim (List[str]): dimensions of the coordinates

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


class Sample(BaseModel):
    """Represents a single sample. It contains data and information related to a single observation or measurement within a dataset."""

    def __init__(
        self,
        directory_path: Union[str, Path] = None,
        mesh_base_name: str = "Base",
        mesh_zone_name: str = "Zone",
        meshes: dict[float, CGNSTree] = None,
        scalars: dict[str, ScalarType] = None,
        time_series: dict[str, TimeSeriesType] = None,
        links: dict[float, list[LinkType]] = None,
        paths: dict[float, list[PathType]] = None,
    ) -> None:
        """Initialize an empty :class:`Sample <plaid.containers.sample.Sample>`.

        Args:
            directory_path (Union[str, Path], optional): The path from which to load PLAID sample files.
            mesh_base_name (str, optional): The base name for the mesh. Defaults to 'Base'.
            mesh_zone_name (str, optional): The zone name for the mesh. Defaults to 'Zone'.
            meshes (dict[float, CGNSTree], optional): A dictionary mapping time steps to CGNSTrees. Defaults to None.
            scalars (dict[str, ScalarType], optional): A dictionary mapping scalar names to their values. Defaults to None.
            time_series (dict[str, TimeSeriesType], optional): A dictionary mapping time series names to their values. Defaults to None.
            links (dict[float, list[LinkType]], optional): A dictionary mapping time steps to lists of links. Defaults to None.
            paths (dict[float, list[PathType]], optional): A dictionary mapping time steps to lists of paths. Defaults to None.

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample

                # 1. Create empty instance of Sample
                sample = Sample()
                print(sample)
                >>> Sample(0 scalars, 0 timestamps, 0 fields, no tree)

                # 2. Load sample  and create Sample instance
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)

        Caution:
            It is assumed that you provided a compatible PLAID sample.
        """
        super().__init__()

        self._mesh_base_name: str = mesh_base_name
        self._mesh_zone_name: str = mesh_zone_name

        self._meshes: dict[float, CGNSTree] = meshes
        self._scalars: dict[str, ScalarType] = scalars
        self._time_series: dict[str, TimeSeriesType] = time_series

        self._links: dict[float, list[LinkType]] = links
        self._paths: dict[float, list[PathType]] = paths

        if directory_path is not None:
            directory_path = Path(directory_path)
            self.load(directory_path)

        self._defaults: dict = {
            "active_base": None,
            "active_zone": None,
            "active_time": None,
        }

        self._extra_data = None

    def copy(self) -> Self:
        """Create a deep copy of the sample.

        Returns:
            A new `Sample` instance with all internal data (scalars, time series, fields, meshes, etc.)
            deeply copied to ensure full isolation from the original.

        Note:
            This operation may be memory-intensive for large samples.
        """
        return copy.deepcopy(self)

    # -------------------------------------------------------------------------#
    def set_default_base(self, base_name: str, time: Optional[float] = None) -> None:
        """Set the default base for the specified time (that will also be set as default if provided).

        The default base is a reference point for various operations in the system.

        Args:
            base_name (str): The name of the base to be set as the default.
            time (float, optional): The time at which the base should be set as default. If not provided, the default base and active zone will be set with the default time.

        Raises:
            ValueError: If the specified base does not exist at the given time.

        Note:
            - Setting the default base and is important for synchronizing operations with a specific base in the system's data.
            - The available mesh base can be obtained using the `get_base_names` method.

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)
                print(sample.get_physical_dim("BaseA", 0.5))
                >>> 3

                # Set "BaseA" as the default base for the default time
                sample.set_default_base("BaseA")

                # You can now use class functions with "BaseA" as default base
                print(sample.get_physical_dim(0.5))
                >>> 3

                # Set "BaseB" as the default base for a specific time
                sample.set_default_base("BaseB", 0.5)

                # You can now use class functions with "BaseB" as default base and 0.5 as default time
                print(sample.get_physical_dim()) # Physical dim of the base "BaseB"
                >>> 3
        """
        if time is not None:
            self.set_default_time(time)
        if base_name in (self._defaults["active_base"], None):
            return
        if not self.has_base(base_name, time):
            raise ValueError(f"base {base_name} does not exist at time {time}")

        self._defaults["active_base"] = base_name

    def set_default_zone_base(
        self, zone_name: str, base_name: str, time: float = None
    ) -> None:
        """Set the default base and active zone for the specified time (that will also be set as default if provided).

        The default base and active zone serve as reference points for various operations in the system.

        Args:
            zone_name (str): The name of the zone to be set as the active zone.
            base_name (str): The name of the base to be set as the default.
            time (float, optional): The time at which the base and zone should be set as default. If not provided, the default base and active zone will be set with the default time.

        Raises:
            ValueError: If the specified base or zone does not exist at the given time

        Note:
            - Setting the default base and zone are important for synchronizing operations with a specific base/zone in the system's data.
            - The available mesh bases and zones can be obtained using the `get_base_names` and `get_base_zones` methods, respectively.

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)
                print(sample.get_zone_type("ZoneX", "BaseA", 0.5))
                >>> Structured

                # Set "BaseA" as the default base and "ZoneX" as the active zone for the default time
                sample.set_default_zone_base("ZoneX", "BaseA")

                # You can now use class functions with "BaseA" as default base with "ZoneX" as default zone
                print(sample.get_zone_type(0.5)) # type of the zone "ZoneX" of base "BaseA"
                >>> Structured

                # Set "BaseB" as the default base and "ZoneY" as the active zone for a specific time
                sample.set_default_zone_base("ZoneY", "BaseB", 0.5)

                # You can now use class functions with "BaseB" as default base with "ZoneY" as default zone and 0.5 as default time
                print(sample.get_zone_type()) # type of the zone "ZoneY" of base "BaseB" at 0.5
                >>> Unstructured
        """
        self.set_default_base(base_name, time)
        if zone_name in (self._defaults["active_zone"], None):
            return
        if not self.has_zone(zone_name, base_name, time):
            raise ValueError(
                f"zone {zone_name} does not exist for the base {base_name} at time {time}"
            )

        self._defaults["active_zone"] = zone_name

    def set_default_time(self, time: float) -> None:
        """Set the default time for the system.

        This function sets the default time to be used for various operations in the system.

        Args:
            time (float): The time value to be set as the default.

        Raises:
            ValueError: If the specified time does not exist in the available mesh times.

        Note:
            - Setting the default time is important for synchronizing operations with a specific time point in the system's data.
            - The available mesh times can be obtained using the `get_all_mesh_times` method.

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)
                print(sample.show_tree(0.5))
                >>> ...

                # Set the default time to 0.5 seconds
                sample.set_default_time(0.5)

                # You can now use class functions with 0.5 as default time
                print(sample.show_tree()) # show the cgns tree at the time 0.5
                >>> ...
        """
        if time in (self._defaults["active_time"], None):
            return
        if time not in self.get_all_mesh_times():
            raise ValueError(f"time {time} does not exist in mesh times")

        self._defaults["active_time"] = time

    def get_time_assignment(self, time: float = None) -> float:
        """Retrieve the default time for the CGNS operations.

        If there are available time steps, it will return the first one; otherwise, it will return 0.0.

        Args:
            time (str, optional): The time value provided for the operation. If not provided, the default time set in the system will be used.

        Returns:
            float: The attributed time.

        Note:
            - The default time step is used as a reference point for many CGNS operations.
            - It is important for accessing and visualizing data at specific time points in a simulation.
        """
        if self._defaults["active_time"] is None and time is None:
            timestamps = self.get_all_mesh_times()
            return sorted(timestamps)[0] if len(timestamps) > 0 else 0.0
        return self._defaults["active_time"] if time is None else time

    def get_base_assignment(self, base_name: str = None, time: float = None) -> str:
        """Retrieve the default base name for the CGNS operations.

        This function calculates the attributed base for a specific operation based on the
        default base set in the system.

        Args:
            base_name (str, optional): The name of the base to attribute the operation to. If not provided, the default base set in the system will be used.
            time (str, optional): The time value provided for the operation. If not provided, the default time set in the system will be used.

        Raises:
            KeyError: If no default base can be determined based on the provided or default.
            KeyError: If no base node is found after following given and default parameters.

        Returns:
            str: The attributed base name.

        Note:
            - If no specific base name is provided, the function will use the default base provided by the user.
            - In case the default base does not exist: If no specific time is provided, the function will use the default time provided by the user.
        """
        base_name = base_name or self._defaults.get("active_base")

        if base_name:
            return base_name

        base_names = self.get_base_names(time=time)
        if len(base_names) == 0:
            return None
        elif len(base_names) == 1:
            # logging.info(f"No default base provided. Taking the only base available: {base_names[0]}")
            return base_names[0]

        raise KeyError(f"No default base provided among {base_names}")

    def get_zone_assignment(
        self, zone_name: str = None, base_name: str = None, time: float = None
    ) -> str:
        """Retrieve the default zone name for the CGNS operations.

        This function calculates the attributed zone for a specific operation based on the
        default zone set in the system, within the specified base.

        Args:
            zone_name (str, optional): The name of the zone to attribute the operation to. If not provided, the default zone set in the system within the specified base will be used.
            base_name (str, optional): The name of the base within which the zone should be attributed. If not provided, the default base set in the system will be used.
            time (str, optional): The time value provided for the operation. If not provided, the default time set in the system will be used.

        Raises:
            KeyError: If no default zone can be determined based on the provided or default values.
            KeyError: If no zone node is found after following given and default parameters.

        Returns:
            str: The attributed zone name.

        Note:
            - If neither a specific zone name nor a specific base name is provided, the function will use the default zone provided by the user.
            - In case the default zone does not exist: If no specific time is provided, the function will use the default time provided by the user.
        """
        zone_name = zone_name or self._defaults.get("active_zone")

        if zone_name:
            return zone_name

        base_name = self.get_base_assignment(base_name, time)
        zone_names = self.get_zone_names(base_name, time=time)
        if len(zone_names) == 0:
            return None
        elif len(zone_names) == 1:
            # logging.info(f"No default zone provided. Taking the only zone available: {zone_names[0]} in default base: {base_name}")
            return zone_names[0]

        raise KeyError(
            f"No default zone provided among {zone_names} in the default base: {base_name}"
        )

    # -------------------------------------------------------------------------#
    def show_tree(self, time: float = None) -> None:
        """Display the structure of the CGNS tree for a specified time.

        Args:
            time (float, optional): The time step for which you want to display the CGNS tree structure. Defaults to None. If a specific time is not provided, the method will display the tree structure for the default time step.

        Examples:
            .. code-block:: python

                # To display the CGNS tree structure for the default time step:
                sample.show_tree()

                # To display the CGNS tree structure for a specific time step:
                sample.show_tree(0.5)
        """
        time = self.get_time_assignment(time)

        if self._meshes is not None:
            CGH.show_cgns_tree(self._meshes[time])

    def init_tree(self, time: float = None) -> CGNSTree:
        """Initialize a CGNS tree structure at a specified time step or create a new one if it doesn't exist.

        Args:
            time (float, optional): The time step for which to initialize the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSTree (list): The initialized or existing CGNS tree structure for the specified time step.
        """
        time = self.get_time_assignment(time)

        if self._meshes is None:
            self._meshes = {time: CGL.newCGNSTree()}
            self._links = {time: None}
            self._paths = {time: None}
        elif time not in self._meshes:
            self._meshes[time] = CGL.newCGNSTree()
            self._links[time] = None
            self._paths[time] = None

        return self._meshes[time]

    def get_mesh(
        self, time: float = None, apply_links: bool = False, in_memory=False
    ) -> CGNSTree:
        """Retrieve the CGNS tree structure for a specified time step, if available.

        Args:
            time (float, optional): The time step for which to retrieve the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.
            apply_links (bool, optional): Activates the following of the CGNS links to reconstruct the complete CGNS tree - in this case, a deepcopy of the tree is made to prevent from modifying the existing tree.
            in_memory (bool, optional): Active if apply_links == True, ONLY WORKING if linked mesh is in the current sample. This option follows the link in memory from current sample.

        Returns:
            CGNSTree: The CGNS tree structure for the specified time step if available; otherwise, returns None.
        """
        if self._meshes is None:
            return None

        time = self.get_time_assignment(time)
        tree = self._meshes[time]

        links = self.get_links(time)
        if not apply_links or links is None:
            return tree

        tree = copy.deepcopy(tree)
        for link in links:
            if not in_memory:
                subtree, _, _ = CGM.load(str(Path(link[0]) / link[1]), subtree=link[2])
            else:
                linked_timestep = int(link[1].split(".cgns")[0].split("_")[1])
                linked_timestamp = list(self._meshes.keys())[linked_timestep]
                subtree = self.get_mesh(linked_timestamp)
            node_path = "/".join(link[2].split("/")[:-1])
            node_to_append = CGU.getNodeByPath(tree, node_path)
            assert node_to_append is not None, (
                f"nodepath {node_path} not present in tree, cannot apply link"
            )
            node_to_append[2].append(CGU.getNodeByPath(subtree, link[2]))

        return tree

    def get_links(self, time: float = None) -> list[LinkType]:
        """Retrieve the CGNS links for a specified time step, if available.

        Args:
            time (float, optional): The time step for which to retrieve the CGNS links. If a specific time is not provided, the method will display the links for the default time step.

        Returns:
            list: The CGNS links for the specified time step if available; otherwise, returns None.
        """
        time = self.get_time_assignment(time)
        return self._links[time] if (self._links is not None) else None

    def get_all_mesh_times(self) -> list[float]:
        """Retrieve all time steps corresponding to the meshes, if available.

        Returns:
            list[float]: A list of all available time steps.
        """
        return list(self._meshes.keys()) if (self._meshes is not None) else []

    def set_meshes(self, meshes: dict[float, CGNSTree]) -> None:
        """Set all meshes with their corresponding time step.

        Args:
            meshes (dict[float,CGNSTree]): Collection of time step with its corresponding CGNSTree.

        Raises:
            KeyError: If there is already a CGNS tree set.
        """
        if self._meshes is None:
            self._meshes = meshes
            self._links = {}
            self._paths = {}
            for time in self._meshes.keys():
                self._links[time] = None
                self._paths[time] = None
        else:
            raise KeyError(
                "meshes is already set, you cannot overwrite it, delete it first or extend it with `Sample.add_tree`"
            )

    def add_tree(self, tree: CGNSTree, time: float = None) -> CGNSTree:
        """Merge a CGNS tree to the already existing tree.

        Args:
            tree (CGNSTree): The CGNS tree to be merged. If a Base node already exists, it is ignored.
            time (float, optional): The time step for which to add the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.

        Raises:
            ValueError: If the provided CGNS tree is an empty list.

        Returns:
            CGNSTree: The merged CGNS tree.
        """
        if tree == []:
            raise ValueError("CGNS Tree should not be an empty list")

        time = self.get_time_assignment(time)

        if self._meshes is None:
            self._meshes = {time: tree}
            self._links = {time: None}
            self._paths = {time: None}
        elif time not in self._meshes:
            self._meshes[time] = tree
            self._links[time] = None
            self._paths[time] = None
        else:
            # TODO: gérer le cas où il y a des bases de mêmes noms... + merge
            # récursif des nœuds
            local_bases = self.get_base_names(time=time)
            base_nodes = CGU.getNodesFromTypeSet(tree, "CGNSBase_t")
            for _, node in base_nodes:
                if node[__NAME__] not in local_bases:  # pragma: no cover
                    self._meshes[time][__CHILDREN__].append(node)
                else:
                    logger.warning(
                        f"base <{node[__NAME__]}> already exists in self._tree --> ignored"
                    )

        base_names = self.get_base_names(time=time)
        for base_name in base_names:
            base_node = self.get_base(base_name, time=time)
            if CGU.getValueByPath(base_node, "Time/TimeValues") is None:
                baseIterativeData_node = CGL.newBaseIterativeData(base_node, "Time", 1)
                TimeValues_node = CGU.newNode(
                    "TimeValues", None, [], CGK.DataArray_ts, baseIterativeData_node
                )
                CGU.setValue(TimeValues_node, np.array([time]))

        return self._meshes[time]

    def del_tree(self, time: float) -> CGNSTree:
        """Delete the CGNS tree for a specific time.

        Args:
            time (float): The time step for which to delete the CGNS tree structure.

        Raises:
            KeyError: There is no CGNS tree in this Sample / There is no CGNS tree for the provided time.

        Returns:
            CGNSTree: The deleted CGNS tree.
        """
        if self._meshes is None:
            raise KeyError("There is no CGNS tree in this sample.")

        if time not in self._meshes:
            raise KeyError(f"There is no CGNS tree for time {time}.")

        self._links.pop(time, None)
        self._paths.pop(time, None)
        return self._meshes.pop(time)

    def link_tree(
        self,
        path_linked_sample: Union[str, Path],
        linked_sample: Self,
        linked_time: float,
        time: float,
    ) -> CGNSTree:
        """Link the geometrical features of the CGNS tree of the current sample at a given time, to the ones of another sample.

        Args:
            path_linked_sample (Union[str, Path]): The absolute path of the folder containing the linked CGNS
            linked_sample (Sample): The linked sample
            linked_time (float): The time step of the linked CGNS in the linked sample
            time (float): The time step the current sample to which the CGNS tree is linked.

        Returns:
            CGNSTree: The deleted CGNS tree.
        """
        # see https://pycgns.github.io/MAP/sids-to-python.html#links
        # difficulty is to link only the geometrical objects, which can be complex

        # https://pycgns.github.io/MAP/examples.html#save-with-links
        # When you load a file all the linked-to files are resolved to produce a full CGNS/Python tree with actual node data.

        path_linked_sample = Path(path_linked_sample)

        if linked_time not in linked_sample._meshes:  # pragma: no cover
            raise KeyError(
                f"There is no CGNS tree for time {linked_time} in linked_sample."
            )
        if time in self._meshes:  # pragma: no cover
            raise KeyError(f"A CGNS tree is already linked in self for time {time}.")

        tree = CGL.newCGNSTree()

        base_names = linked_sample.get_base_names(time=linked_time)

        for bn in base_names:
            base_node = linked_sample.get_base(bn, time=linked_time)
            base = [bn, base_node[1], [], "CGNSBase_t"]
            tree[2].append(base)

            family = [
                "Bulk",
                np.array([b"B", b"u", b"l", b"k"], dtype="|S1"),
                [],
                "FamilyName_t",
            ]  # maybe get this from linked_sample as well ?
            base[2].append(family)

            zone_names = linked_sample.get_zone_names(bn, time=linked_time)
            for zn in zone_names:
                zone_node = linked_sample.get_zone(zn, bn, time=linked_time)
                grid = [
                    zn,
                    zone_node[1],
                    [
                        [
                            "ZoneType",
                            np.array(
                                [
                                    b"U",
                                    b"n",
                                    b"s",
                                    b"t",
                                    b"r",
                                    b"u",
                                    b"c",
                                    b"t",
                                    b"u",
                                    b"r",
                                    b"e",
                                    b"d",
                                ],
                                dtype="|S1",
                            ),
                            [],
                            "ZoneType_t",
                        ]
                    ],
                    "Zone_t",
                ]
                base[2].append(grid)
                zone_family = [
                    "FamilyName",
                    np.array([b"B", b"u", b"l", b"k"], dtype="|S1"),
                    [],
                    "FamilyName_t",
                ]
                grid[2].append(zone_family)

        def find_feature_roots(sample: Sample, time: float, Type_t: str):
            Types_t = CGU.getAllNodesByTypeSet(sample.get_mesh(time), Type_t)
            # in case the type is not present in the tree
            if Types_t == []:  # pragma: no cover
                return []
            types = [Types_t[0]]
            for t in Types_t[1:]:
                for tt in types:
                    if tt not in t:  # pragma: no cover
                        types.append(t)
            return types

        feature_paths = []
        for feature in ["ZoneBC_t", "Elements_t", "GridCoordinates_t"]:
            feature_paths += find_feature_roots(linked_sample, linked_time, feature)

        self.add_tree(tree, time=time)

        dname = path_linked_sample.parent
        bname = path_linked_sample.name
        self._links[time] = [[str(dname), bname, fp, fp] for fp in feature_paths]

        return tree

    # -------------------------------------------------------------------------#
    def get_topological_dim(self, base_name: str = None, time: float = None) -> int:
        """Get the topological dimension of a base node at a specific time.

        Args:
            base_name (str, optional): The name of the base node for which to retrieve the topological dimension. Defaults to None.
            time (float, optional): The time at which to retrieve the topological dimension. Defaults to None.

        Raises:
            ValueError: If there is no base node with the specified `base_name` at the given `time` in this sample.

        Returns:
            int: The topological dimension of the specified base node at the given time.
        """
        # get_base will look for default time and base_name
        base_node = self.get_base(base_name, time)

        if base_node is None:  # pragma: no cover
            raise ValueError(
                f"there is no base called {base_name} at the time {time} in this sample"
            )

        return base_node[1][0]

    def get_physical_dim(self, base_name: str = None, time: float = None) -> int:
        """Get the physical dimension of a base node at a specific time.

        Args:
            base_name (str, optional): The name of the base node for which to retrieve the topological dimension. Defaults to None.
            time (float, optional): The time at which to retrieve the topological dimension. Defaults to None.

        Raises:
            ValueError: If there is no base node with the specified `base_name` at the given `time` in this sample.

        Returns:
            int: The topological dimension of the specified base node at the given time.
        """
        base_node = self.get_base(base_name, time)
        if base_node is None:  # pragma: no cover
            raise ValueError(
                f"there is no base called {base_name} at the time {time} in this sample"
            )

        return base_node[1][1]

    def init_base(
        self,
        topological_dim: int,
        physical_dim: int,
        base_name: str = None,
        time: float = None,
    ) -> CGNSNode:
        """Create a Base node named `base_name` if it doesn't already exists.

        Args:
            topological_dim (int): Cell dimension, see [CGNS standard](https://pycgns.github.io/PAT/lib.html#CGNS.PAT.cgnslib.newCGNSBase).
            physical_dim (int): Ambient space dimension, see [CGNS standard](https://pycgns.github.io/PAT/lib.html#CGNS.PAT.cgnslib.newCGNSBase).
            base_name (str): If not specified, uses `mesh_base_name` specified in Sample initialization. Defaults to None.
            time (float, optional): The time at which to initialize the base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSNode: The created Base node.
        """
        _check_names([base_name])

        time = self.get_time_assignment(time)

        if base_name is None:
            base_name = (
                self._mesh_base_name
                + "_"
                + str(topological_dim)
                + "_"
                + str(physical_dim)
            )

        self.init_tree(time)
        if not (self.has_base(base_name, time)):
            base_node = CGL.newCGNSBase(
                self._meshes[time], base_name, topological_dim, physical_dim
            )

        base_names = self.get_base_names(time=time)
        for base_name in base_names:
            base_node = self.get_base(base_name, time=time)
            if CGU.getValueByPath(base_node, "Time/TimeValues") is None:
                base_iterative_data_node = CGL.newBaseIterativeData(
                    base_node, "Time", 1
                )
                time_values_node = CGU.newNode(
                    "TimeValues", None, [], CGK.DataArray_ts, base_iterative_data_node
                )
                CGU.setValue(time_values_node, np.array([time]))

        return base_node

    def del_base(self, base_name: str, time: float) -> CGNSTree:
        """Delete a CGNS base node for a specific time.

        Args:
            base_name (str): The name of the base node to be deleted.
            time (float): The time step for which to delete the CGNS base node.

        Raises:
            KeyError: There is no CGNS tree in this sample / There is no CGNS tree for the provided time.
            KeyError: If there is no base node with the given base name or time.

        Returns:
            CGNSTree: The tree at the provided time (without the deleted node)
        """
        if self._meshes is None:
            raise KeyError("There is no CGNS tree in this sample.")

        if time not in self._meshes:
            raise KeyError(f"There is no CGNS tree for time {time}.")

        base_node = self.get_base(base_name, time)
        mesh_tree = self._meshes[time]

        if base_node is None:
            raise KeyError(
                f"There is no base node with name {base_name} for time {time}."
            )

        return CGU.nodeDelete(mesh_tree, base_node)

    def get_base_names(
        self, full_path: bool = False, unique: bool = False, time: float = None
    ) -> list[str]:
        """Return Base names.

        Args:
            full_path (bool, optional): If True, returns full paths instead of only Base names. Defaults to False.
            unique (bool, optional): If True, returns unique names instead of potentially duplicated names. Defaults to False.
            time (float, optional): The time at which to check for the Base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            list[str]:
        """
        time = self.get_time_assignment(time)

        if self._meshes is not None:
            if self._meshes[time] is not None:
                return CGH.get_base_names(
                    self._meshes[time], full_path=full_path, unique=unique
                )
        else:
            return []

    def has_base(self, base_name: str, time: float = None) -> bool:
        """Check if a CGNS tree contains a Base with a given name at a specified time.

        Args:
            base_name (str): The name of the Base to check for in the CGNS tree.
            time (float, optional): The time at which to check for the Base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            bool: `True` if the CGNS tree has a Base called `base_name`, else return `False`.
        """
        # get_base_names will look for the default time
        return base_name in self.get_base_names(time=time)

    def get_base(self, base_name: str = None, time: float = None) -> CGNSNode:
        """Return Base node named `base_name`.

        If `base_name` is not specified, checks that there is **at most** one base, else raises an error.

        Args:
            base_name (str, optional): The name of the Base node to retrieve. Defaults to None. Defaults to None.
            time (float, optional): Time at which you want to retrieve the Base node. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSNode or None: The Base node with the specified name or None if it is not found.
        """
        time = self.get_time_assignment(time)
        base_name = self.get_base_assignment(base_name, time)

        if (self._meshes is None) or (self._meshes[time] is None):
            logger.warning(f"No base with name {base_name} and this tree")
            return None

        return CGU.getNodeByPath(self._meshes[time], f"/CGNSTree/{base_name}")

    # -------------------------------------------------------------------------#
    def init_zone(
        self,
        zone_shape: np.ndarray,
        zone_type: str = CGK.Unstructured_s,
        zone_name: str = None,
        base_name: str = None,
        time: float = None,
    ) -> CGNSNode:
        """Initialize a new zone within a CGNS base.

        Args:
            zone_shape (np.ndarray): An array specifying the shape or dimensions of the zone.
            zone_type (str, optional): The type of the zone. Defaults to CGK.Unstructured_s.
            zone_name (str, optional): The name of the zone to initialize. If not provided, uses `mesh_zone_name` specified in Sample initialization. Defaults to None.
            base_name (str, optional): The name of the base to which the zone will be added. If not provided, the zone will be added to the currently active base. Defaults to None.
            time (float, optional): The time at which to initialize the zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Raises:
            KeyError: If the specified base does not exist. You can create a base using `Sample.init_base(base_name)`.

        Returns:
            CGLNode: The newly initialized zone node within the CGNS tree.
        """
        _check_names([zone_name])

        # init_tree will look for default time
        self.init_tree(time)
        # get_base will look for default base_name and time
        base_node = self.get_base(base_name, time)
        if base_node is None:
            raise KeyError(
                f"there is no base <{base_name}>, you should first create one with `Sample.init_base({base_name=})`"
            )

        zone_name = self.get_zone_assignment(zone_name, base_name, time)
        if zone_name is None:
            zone_name = self._mesh_zone_name

        zone_node = CGL.newZone(base_node, zone_name, zone_shape, zone_type)
        return zone_node

    def del_zone(self, zone_name: str, base_name: str, time: float) -> CGNSTree:
        """Delete a zone within a CGNS base.

        Args:
            zone_name (str): The name of the zone to be deleted.
            base_name (str, optional): The name of the base from which the zone will be deleted. If not provided, the zone will be deleted from the currently active base. Defaults to None.
            time (float, optional): The time step for which to delete the zone. Defaults to None.

        Raises:
            KeyError: There is no CGNS tree in this sample / There is no CGNS tree for the provided time.
            KeyError: If there is no base node with the given base name or time.

        Returns:
            CGNSTree: The tree at the provided time (without the deleted node)
        """
        if self._meshes is None:  # pragma: no cover
            raise KeyError("There is no CGNS tree in this sample.")

        if time not in self._meshes:
            raise KeyError(f"There is no CGNS tree for time {time}.")

        zone_node = self.get_zone(zone_name, base_name, time)
        mesh_tree = self._meshes[time]

        if zone_node is None:
            raise KeyError(
                f"There is no zone node with name {zone_name} or base node with name {base_name}."
            )

        return CGU.nodeDelete(mesh_tree, zone_node)

    def get_zone_names(
        self,
        base_name: str = None,
        full_path: bool = False,
        unique: bool = False,
        time: float = None,
    ) -> list[str]:
        """Return list of Zone names in Base named `base_name` with specific time.

        Args:
            base_name (str, optional): Name of Base where to search Zones. If not specified, checks if there is at most one Base. Defaults to None.
            full_path (bool, optional): If True, returns full paths instead of only Zone names. Defaults to False.
            unique (bool, optional): If True, returns unique names instead of potentially duplicated names. Defaults to False.
            time (float, optional): The time at which to check for the Zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            list[str]: List of Zone names in Base named `base_name`, empty if there is none or if the Base doesn't exist.
        """
        zone_paths = []

        # get_base will look for default base_name and time
        base_node = self.get_base(base_name, time)
        if base_node is not None:
            z_paths = CGU.getPathsByTypeSet(base_node, "CGNSZone_t")
            for pth in z_paths:
                s_pth = pth.split("/")
                assert len(s_pth) == 2
                assert s_pth[0] == base_name or base_name is None
                if full_path:
                    zone_paths.append(pth)
                else:
                    zone_paths.append(s_pth[1])

        if unique:
            return list(set(zone_paths))
        else:
            return zone_paths

    def has_zone(
        self, zone_name: str, base_name: str = None, time: float = None
    ) -> bool:
        """Check if the CGNS tree contains a Zone with the specified name within a specific Base and time.

        Args:
            zone_name (str): The name of the Zone to check for.
            base_name (str, optional): The name of the Base where the Zone should be located. If not provided, the function checks all bases. Defaults to None.
            time (float, optional): The time at which to check for the Zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            bool: `True` if the CGNS tree has a Zone called `zone_name` in a Base called `base_name`, else return `False`.
        """
        # get_zone_names will look for default base_name and time
        return zone_name in self.get_zone_names(base_name, time=time)

    def get_zone(
        self, zone_name: str = None, base_name: str = None, time: float = None
    ) -> CGNSNode:
        """Retrieve a CGNS Zone node by its name within a specific Base and time.

        Args:
            zone_name (str, optional): The name of the Zone node to retrieve. If not specified, checks that there is **at most** one zone in the base, else raises an error. Defaults to None.
            base_name (str, optional): The Base in which to seek to zone retrieve. If not specified, checks that there is **at most** one base, else raises an error. Defaults to None.
            time (float, optional): Time at which you want to retrieve the Zone node.

        Returns:
            CGNSNode: Returns a CGNS Zone node if found; otherwise, returns None.
        """
        # get_base will look for default base_name and time
        base_node = self.get_base(base_name, time)
        if base_node is None:
            logger.warning(f"No base with name {base_name} and this tree")
            return None

        # _zone_attribution will look for default base_name
        zone_name = self.get_zone_assignment(zone_name, base_name, time)
        if zone_name is None:
            logger.warning(f"No zone with name {zone_name} and this base ({base_name})")
            return None

        return CGU.getNodeByPath(base_node, zone_name)

    def get_zone_type(
        self, zone_name: str = None, base_name: str = None, time: float = None
    ) -> str:
        """Get the type of a specific zone at a specified time.

        Args:
            zone_name (str, optional): The name of the zone whose type you want to retrieve. Default is None.
            base_name (str, optional): The name of the base in which the zone is located. Default is None.
            time (float, optional): The timestamp for which you want to retrieve the zone type. Default is 0.0.

        Raises:
            KeyError: Raised when the specified zone or base does not exist. You should first create the base/zone using `Sample.init_zone(zone_name, base_name)`.

        Returns:
            str: The type of the specified zone as a string.
        """
        # get_zone will look for default base_name, zone_name and time
        zone_node = self.get_zone(zone_name, base_name, time)

        if zone_node is None:
            raise KeyError(
                f"there is no base/zone <{base_name}/{zone_name}>, you should first create one with `Sample.init_zone({zone_name=},{base_name=})`"
            )
        return CGU.getValueByPath(zone_node, "ZoneType").tobytes().decode()

    # -------------------------------------------------------------------------#
    def get_scalar_names(self) -> set[str]:
        """Get a set of scalar names available in the object.

        Returns:
            set[str]: A set containing the names of the available scalars.
        """
        if self._scalars is None:
            return []
        else:
            res = sorted(self._scalars.keys())
            return res

    def get_scalar(self, name: str) -> ScalarType:
        """Retrieve a scalar value associated with the given name.

        Args:
            name (str): The name of the scalar value to retrieve.

        Returns:
            ScalarType or None: The scalar value associated with the given name, or None if the name is not found.
        """
        if (self._scalars is None) or (name not in self._scalars):
            return None
        else:
            return self._scalars[name]

    def add_scalar(self, name: str, value: ScalarType) -> None:
        """Add a scalar value to a dictionary.

        Args:
            name (str): The name of the scalar value.
            value (ScalarType): The scalar value to add or update in the dictionary.
        """
        _check_names([name])
        if self._scalars is None:
            self._scalars = {name: value}
        else:
            self._scalars[name] = value

    def del_scalar(self, name: str) -> ScalarType:
        """Delete a scalar value from the dictionary.

        Args:
            name (str): The name of the scalar value to be deleted.

        Raises:
            KeyError: Raised when there is no scalar / there is no scalar with the provided name.

        Returns:
            ScalarType: The value of the deleted scalar.
        """
        if self._scalars is None:
            raise KeyError("There is no scalar inside this sample.")

        if name not in self._scalars:
            raise KeyError(f"There is no scalar value with name {name}.")

        return self._scalars.pop(name)

    # -------------------------------------------------------------------------#
    def get_time_series_names(self) -> set[str]:
        """Get the names of time series associated with the object.

        Returns:
            set[str]: A set of strings containing the names of the time series.
        """
        if self._time_series is None:
            return []
        else:
            return list(self._time_series.keys())

    def get_time_series(self, name: str) -> TimeSeriesType:
        """Retrieve a time series by name.

        Args:
            name (str): The name of the time series to retrieve.

        Returns:
            TimeSeriesType or None: If a time series with the given name exists, it returns the corresponding time series, or None otherwise.

        """
        if (self._time_series is None) or (name not in self._time_series):
            return None
        else:
            return self._time_series[name]

    def add_time_series(
        self, name: str, time_sequence: TimeSequenceType, values: FieldType
    ) -> None:
        """Add a time series to the sample.

        Args:
            name (str): A descriptive name for the time series.
            time_sequence (TimeSequenceType): The time sequence, array of time points.
            values (FieldType): The values corresponding to the time sequence.

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample.add_time_series('stuff', np.arange(2), np.random.randn(2))
                print(sample.get_time_series('stuff'))
                >>> (array([0, 1]), array([-0.59630135, -1.15572306]))

        Raises:
            TypeError: Raised if the length of `time_sequence` is not equal to the length of `values`.
        """
        _check_names([name])
        assert len(time_sequence) == len(values), (
            "time sequence and values do not have the same size"
        )
        if self._time_series is None:
            self._time_series = {name: (time_sequence, values)}
        else:
            self._time_series[name] = (time_sequence, values)

    def del_time_series(self, name: str) -> tuple[TimeSequenceType, FieldType]:
        """Delete a time series from the sample.

        Args:
            name (str): The name of the time series to be deleted.

        Raises:
            KeyError: Raised when there is no time series / there is no time series with the provided name.

        Returns:
            Tuple[TimeSequenceType, FieldType]: A tuple containing the time sequence and values of the deleted time series.
        """
        if self._time_series is None:
            raise KeyError("There is no time series inside this sample.")

        if name not in self._time_series:
            raise KeyError(f"There is no time series with name {name}.")

        return self._time_series.pop(name)

    # -------------------------------------------------------------------------#
    def get_nodal_tags(
        self, zone_name: str = None, base_name: str = None, time: float = None
    ) -> dict[str, np.ndarray]:
        """Get the nodal tags for a specified base and zone at a given time.

        Args:
            zone_name (str, optional): The name of the zone for which element connectivity data is requested. Defaults to None, indicating the default zone.
            base_name (str, optional): The name of the base for which element connectivity data is requested. Defaults to None, indicating the default base.
            time (float, optional): The time at which element connectivity data is requested. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            dict[str,np.ndarray]: A dictionary where keys are nodal tags names and values are NumPy arrays containing the corresponding tag indices.
            The NumPy arrays have shape (num_nodal_tags).
        """
        # get_zone will look for default base_name, zone_name and time
        zone_node = self.get_zone(zone_name, base_name, time)

        if zone_node is None:
            return {}

        nodal_tags = {}

        gridCoordinatesPath = CGU.getPathsByTypeSet(zone_node, ["GridCoordinates_t"])[0]
        gx = CGU.getNodeByPath(zone_node, gridCoordinatesPath + "/CoordinateX")[1]
        dim = gx.shape

        BCPaths = CGU.getPathsByTypeList(zone_node, ["Zone_t", "ZoneBC_t", "BC_t"])

        for BCPath in BCPaths:
            BCNode = CGU.getNodeByPath(zone_node, BCPath)
            BCName = BCNode[0]
            indices = read_index(BCNode, dim)
            if len(indices) == 0:  # pragma: no cover
                continue

            gl = CGU.getPathsByTypeSet(BCNode, ["GridLocation_t"])
            if gl:
                location = CGU.getValueAsString(CGU.getNodeByPath(BCNode, gl[0]))
            else:  # pragma: no cover
                location = "Vertex"
            if location == "Vertex":
                nodal_tags[BCName] = indices - 1

        ZSRPaths = CGU.getPathsByTypeList(zone_node, ["Zone_t", "ZoneSubRegion_t"])
        for path in ZSRPaths:  # pragma: no cover
            ZSRNode = CGU.getNodeByPath(zone_node, path)
            # fnpath = CGU.getPathsByTypeList(
            #     ZSRNode, ["ZoneSubRegion_t", "FamilyName_t"]
            # )
            # if fnpath:
            #     fn = CGU.getNodeByPath(ZSRNode, fnpath[0])
            #     familyName = CGU.getValueAsString(fn)
            indices = read_index(ZSRNode, dim)
            if len(indices) == 0:
                continue
            gl = CGU.getPathsByTypeSet(ZSRNode, ["GridLocation_t"])[0]
            location = CGU.getValueAsString(CGU.getNodeByPath(ZSRNode, gl))
            if not gl or location == "Vertex":
                nodal_tags[BCName] = indices - 1

        sorted_nodal_tags = {key: np.sort(value) for key, value in nodal_tags.items()}

        return sorted_nodal_tags

    # -------------------------------------------------------------------------#
    def get_nodes(
        self, zone_name: str = None, base_name: str = None, time: float = None
    ) -> Optional[np.ndarray]:
        """Get grid node coordinates from a specified base, zone, and time.

        Args:
            zone_name (str, optional): The name of the zone to search for. Defaults to None.
            base_name (str, optional): The name of the base to search for. Defaults to None.
            time (float, optional):  The time value to consider when searching for the zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Raises:
            TypeError: Raised if multiple <GridCoordinates> nodes are found. Only one is expected.

        Returns:
            Optional[np.ndarray]: A NumPy array containing the grid node coordinates.
            If no matching zone or grid coordinates are found, None is returned.

        Seealso:
            This function can also be called using `get_points()` or `get_vertices()`.
        """
        # get_zone will look for default base_name, zone_name and time
        search_node = self.get_zone(zone_name, base_name, time)

        if search_node is None:
            return None

        grid_paths = CGU.getAllNodesByTypeSet(search_node, ["GridCoordinates_t"])
        if len(grid_paths) == 1:
            grid_node = CGU.getNodeByPath(search_node, grid_paths[0])
            array_x = CGU.getValueByPath(grid_node, "GridCoordinates/CoordinateX")
            array_y = CGU.getValueByPath(grid_node, "GridCoordinates/CoordinateY")
            array_z = CGU.getValueByPath(grid_node, "GridCoordinates/CoordinateZ")
            if array_z is None:
                array = np.concatenate(
                    (array_x.reshape((-1, 1)), array_y.reshape((-1, 1))), axis=1
                )
            else:
                array = np.concatenate(
                    (
                        array_x.reshape((-1, 1)),
                        array_y.reshape((-1, 1)),
                        array_z.reshape((-1, 1)),
                    ),
                    axis=1,
                )
            return array
        elif len(grid_paths) > 1:  # pragma: no cover
            raise TypeError(
                f"Found {len(grid_paths)} <GridCoordinates> nodes, should find only one"
            )

    get_points = get_nodes
    get_vertices = get_nodes

    def set_nodes(
        self,
        nodes: np.ndarray,
        zone_name: str = None,
        base_name: str = None,
        time: float = None,
    ) -> None:
        """Set the coordinates of nodes for a specified base and zone at a given time.

        Args:
            nodes (np.ndarray): A numpy array containing the new node coordinates.
            zone_name (str, optional): The name of the zone where the nodes should be updated. Defaults to None.
            base_name (str, optional): The name of the base where the nodes should be updated. Defaults to None.
            time (float, optional): The time at which the node coordinates should be updated. If a specific time is not provided, the method will display the tree structure for the default time step.

        Raises:
            KeyError: Raised if the specified base or zone do not exist. You should first
            create the base and zone using the `Sample.init_zone(zone_name,base_name)` method.

        Seealso:
            This function can also be called using `set_points()` or `set_vertices()`
        """
        # get_zone will look for default base_name, zone_name and time
        zone_node = self.get_zone(zone_name, base_name, time)

        if zone_node is None:
            raise KeyError(
                f"there is no base/zone <{base_name}/{zone_name}>, you should first create one with `Sample.init_zone({zone_name=},{base_name=})`"
            )

        # Check if GridCoordinates_t node exists
        gc_nodes = [
            child for child in zone_node[2] if child[0] in CGK.GridCoordinates_ts
        ]
        if gc_nodes:
            grid_coords_node = gc_nodes[0]

        coord_type = [CGK.CoordinateX_s, CGK.CoordinateY_s, CGK.CoordinateZ_s]
        for i_dim in range(nodes.shape[-1]):
            name = coord_type[i_dim]

            # Remove existing coordinate if present
            if gc_nodes:
                grid_coords_node[2] = [
                    child for child in grid_coords_node[2] if child[0] != name
                ]

            # Create new coordinate
            CGL.newCoordinates(zone_node, name, np.asfortranarray(nodes[..., i_dim]))

    set_points = set_nodes
    set_vertices = set_nodes

    # -------------------------------------------------------------------------#
    def get_elements(
        self, zone_name: str = None, base_name: str = None, time: float = None
    ) -> dict[str, np.ndarray]:
        """Retrieve element connectivity data for a specified zone, base, and time.

        Args:
            zone_name (str, optional): The name of the zone for which element connectivity data is requested. Defaults to None, indicating the default zone.
            base_name (str, optional): The name of the base for which element connectivity data is requested. Defaults to None, indicating the default base.
            time (float, optional): The time at which element connectivity data is requested. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            dict[str,np.ndarray]: A dictionary where keys are element type names and values are NumPy arrays representing the element connectivity data.
            The NumPy arrays have shape (num_elements, num_nodes_per_element), and element indices are 0-based.
        """
        # get_zone will look for default base_name, zone_name and time
        zone_node = self.get_zone(zone_name, base_name, time)

        if zone_node is None:
            return {}

        elements = {}
        elem_paths = CGU.getAllNodesByTypeSet(zone_node, ["Elements_t"])

        for elem in elem_paths:
            elem_node = CGU.getNodeByPath(zone_node, elem)
            val = CGU.getValue(elem_node)
            elem_type = CGNS_ELEMENT_NAMES[val[0]]
            elem_size = int(elem_type.split("_")[-1])
            # elem_range = CGU.getValueByPath(
            #     elem_node, "ElementRange"
            # )  # TODO elem_range is unused
            # -1 is to get back indexes starting at 0
            elements[elem_type] = (
                CGU.getValueByPath(elem_node, "ElementConnectivity").reshape(
                    (-1, elem_size)
                )
                - 1
            )

        return elements

    # -------------------------------------------------------------------------#
    def get_field_names(
        self,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
    ) -> set[str]:
        """Get a set of field names associated with a specified zone, base, location, and time.

        Args:
            zone_name (str, optional): The name of the zone to search for. Defaults to None.
            base_name (str, optional): The name of the base to search for. Defaults to None.
            location (str, optional): The desired grid location where the field is defined. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            time (float, optional): The specific time at which to retrieve field names. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            set[str]: A set containing the names of the fields that match the specified criteria.
        """

        def get_field_names_one_base(base_name: str) -> list[str]:
            # get_zone will look for default zone_name, base_name, time
            search_node = self.get_zone(zone_name, base_name, time)
            if search_node is None:  # pragma: no cover
                return []

            names = []
            solution_paths = CGU.getPathsByTypeSet(search_node, [CGK.FlowSolution_t])
            for f_path in solution_paths:
                if (
                    CGU.getValueByPath(search_node, f_path + "/GridLocation")
                    .tobytes()
                    .decode()
                    != location
                ):
                    continue
                f_node = CGU.getNodeByPath(search_node, f_path)
                for path in CGU.getPathByTypeFilter(f_node, CGK.DataArray_t):
                    field_name = path.split("/")[-1]
                    if not (field_name == "GridLocation"):
                        names.append(field_name)
            return names

        if base_name is None:
            # get_base_names will look for default time
            base_names = self.get_base_names(time=time)
        else:
            base_names = [base_name]

        all_names = []
        for bn in base_names:
            all_names += get_field_names_one_base(bn)

        all_names.sort()
        all_names = list(set(all_names))

        return all_names

    def get_field(
        self,
        name: str,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
    ) -> FieldType:
        """Retrieve a field with a specified name from a given zone, base, location, and time.

        Args:
            name (str): The name of the field to retrieve.
            zone_name (str, optional): The name of the zone to search for. Defaults to None.
            base_name (str, optional): The name of the base to search for. Defaults to None.
            location (str, optional): The location at which to retrieve the field. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            time (float, optional): The time value to consider when searching for the field. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            FieldType: A set containing the names of the fields that match the specified criteria.
        """
        # get_zone will look for default time
        search_node = self.get_zone(zone_name, base_name, time)
        if search_node is None:
            return None

        is_empty = True
        full_field = []

        solution_paths = CGU.getPathsByTypeSet(search_node, [CGK.FlowSolution_t])

        for f_path in solution_paths:
            if (
                CGU.getValueByPath(search_node, f_path + "/GridLocation")
                .tobytes()
                .decode()
                == location
            ):
                field = CGU.getValueByPath(search_node, f_path + "/" + name)

                if field is None:
                    field = np.empty((0,))
                else:
                    is_empty = False
                full_field.append(field)

        if is_empty:
            return None
        else:
            return np.concatenate(full_field)

    def add_field(
        self,
        name: str,
        field: FieldType,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
        warning_overwrite=True,
    ) -> None:
        """Add a field to a specified zone in the grid.

        Args:
            name (str): The name of the field to be added.
            field (FieldType): The field data to be added.
            zone_name (str, optional): The name of the zone where the field will be added. Defaults to None.
            base_name (str, optional): The name of the base where the zone is located. Defaults to None.
            location (str, optional): The grid location where the field will be stored. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            time (float, optional): The time associated with the field. Defaults to 0.
            warning_overwrite (bool, optional): Show warning if an preexisting field is being overwritten

        Raises:
            KeyError: Raised if the specified zone does not exist in the given base.
        """
        _check_names([name])
        # init_tree will look for default time
        self.init_tree(time)
        # get_zone will look for default zone_name, base_name and time
        zone_node = self.get_zone(zone_name, base_name, time)

        if zone_node is None:
            raise KeyError(
                f"there is no Zone with name {zone_name} in base {base_name}. Did you check topological and physical dimensions ?"
            )

        # solution_paths = CGU.getPathsByTypeOrNameList(self._tree, '/.*/.*/FlowSolution_t')
        solution_paths = CGU.getPathsByTypeSet(zone_node, "FlowSolution_t")
        has_FlowSolution_with_location = False
        if len(solution_paths) > 0:
            for s_path in solution_paths:
                val_location = (
                    CGU.getValueByPath(zone_node, f"{s_path}/GridLocation")
                    .tobytes()
                    .decode()
                )
                if val_location == location:
                    has_FlowSolution_with_location = True

        if not (has_FlowSolution_with_location):
            CGL.newFlowSolution(zone_node, f"{location}Fields", gridlocation=location)

        solution_paths = CGU.getPathsByTypeSet(zone_node, "FlowSolution_t")
        assert len(solution_paths) > 0

        for s_path in solution_paths:
            val_location = (
                CGU.getValueByPath(zone_node, f"{s_path}/GridLocation")
                .tobytes()
                .decode()
            )

            if val_location != location:
                continue

            field_node = CGU.getNodeByPath(zone_node, f"{s_path}/{name}")

            if field_node is None:
                flow_solution_node = CGU.getNodeByPath(zone_node, s_path)
                # CGL.newDataArray(flow_solution_node, name, np.asfortranarray(np.copy(field), dtype=np.float64))
                CGL.newDataArray(flow_solution_node, name, np.asfortranarray(field))
                # res =  [name, np.asfortranarray(field, dtype=np.float32), [], 'DataArray_t']
                # print(field.shape)
                # flow_solution_node[2].append(res)
            else:
                if warning_overwrite:
                    logger.warning(
                        f"field node with name {name} already exists -> data will be replaced"
                    )
                CGU.setValue(field_node, np.asfortranarray(field))

    def del_field(
        self,
        name: str,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
    ) -> CGNSTree:
        """Delete a field from a specified zone in the grid.

        Args:
            name (str): The name of the field to be deleted.
            zone_name (str, optional): The name of the zone from which the field will be deleted. Defaults to None.
            base_name (str, optional): The name of the base where the zone is located. Defaults to None.
            location (str, optional): The grid location where the field is stored. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            time (float, optional): The time associated with the field. Defaults to 0.

        Raises:
            KeyError: Raised if the specified zone or field does not exist in the given base.

        Returns:
            CGNSTree: The tree at the provided time (without the deleted node)
        """
        # get_zone will look for default zone_name, base_name, and time
        zone_node = self.get_zone(zone_name, base_name, time)
        time = self.get_time_assignment(time)
        mesh_tree = self._meshes[time]

        if zone_node is None:
            raise KeyError(
                f"There is no Zone with name {zone_name} in base {base_name}."
            )

        solution_paths = CGU.getPathsByTypeSet(zone_node, [CGK.FlowSolution_t])

        updated_tree = None
        for s_path in solution_paths:
            if (
                CGU.getValueByPath(zone_node, f"{s_path}/GridLocation")
                .tobytes()
                .decode()
                == location
            ):
                field_node = CGU.getNodeByPath(zone_node, f"{s_path}/{name}")
                if field_node is not None:
                    updated_tree = CGU.nodeDelete(mesh_tree, field_node)

        # If the function reaches here, the field was not found
        if updated_tree is None:
            raise KeyError(f"There is no field with name {name} in the specified zone.")

        return updated_tree

    def del_all_fields(
        self,
    ) -> Self:
        """Delete alls field from sample, while keeping geometrical info.

        Returns:
            Sample: The sample with deleted fields
        """
        all_features_identifiers = self.get_all_features_identifiers()
        # Delete all fields in the sample
        for feat_id in all_features_identifiers:
            if feat_id["type"] == "field":
                self.del_field(
                    name=feat_id["name"],
                    zone_name=feat_id["zone_name"],
                    base_name=feat_id["base_name"],
                    location=feat_id["location"],
                    time=feat_id["time"],
                )
        return self

    # -------------------------------------------------------------------------#
    def get_all_features_identifiers(
        self,
    ) -> list[FeatureIdentifier]:
        """Get all features identifiers from the sample.

        Returns:
            list[FeatureIdentifier]: A list of dictionaries containing the identifiers of all features in the sample.
        """
        all_features_identifiers = []
        for sn in self.get_scalar_names():
            all_features_identifiers.append({"type": "scalar", "name": sn})
        for tsn in self.get_time_series_names():
            all_features_identifiers.append({"type": "time_series", "name": tsn})
        for t in self.get_all_mesh_times():
            for bn in self.get_base_names(time=t):
                for zn in self.get_zone_names(base_name=bn, time=t):
                    if self.get_nodes(base_name=bn, zone_name=zn, time=t) is not None:
                        all_features_identifiers.append(
                            {
                                "type": "nodes",
                                "base_name": bn,
                                "zone_name": zn,
                                "time": t,
                            }
                        )
                    for loc in CGNS_FIELD_LOCATIONS:
                        for fn in self.get_field_names(
                            zone_name=zn, base_name=bn, location=loc, time=t
                        ):
                            all_features_identifiers.append(
                                {
                                    "type": "field",
                                    "name": fn,
                                    "base_name": bn,
                                    "zone_name": zn,
                                    "location": loc,
                                    "time": t,
                                }
                            )
        return all_features_identifiers

    def get_all_features_identifiers_by_type(
        self, feature_type: str
    ) -> list[FeatureIdentifier]:
        """Get all features identifiers of a given type from the sample.

        Args:
            feature_type (str): Type of features to return

        Returns:
            list[FeatureIdentifier]: A list of dictionaries containing the identifiers of a given type of all features in the sample.
        """
        assert feature_type in AUTHORIZED_FEATURE_TYPES, "feature_type not known"
        all_features_identifiers = self.get_all_features_identifiers()
        return [
            feat_id
            for feat_id in all_features_identifiers
            if feat_id["type"] == feature_type
        ]

    def get_feature_from_string_identifier(
        self, feature_string_identifier: str
    ) -> FeatureType:
        """Retrieve a specific feature from its encoded string identifier.

        The `feature_string_identifier` must follow the format:
            "<feature_type>::<detail1>/<detail2>/.../<detailN>"

        Supported feature types:
            - "scalar": expects 1 detail → `get_scalar(name)`
            - "time_series": expects 1 detail → `get_time_series(name)`
            - "field": up to 5 details → `get_field(name, base_name, zone_name, location, time)`
            - "nodes": up to 3 details → `get_nodes(base_name, zone_name, time)`

        Args:
            feature_string_identifier (str): Structured identifier of the feature.

        Returns:
            FeatureType: The retrieved feature object.

        Raises:
            AssertionError: If `feature_type` is unknown.

        Warnings:
            - If "time" is present in a field/nodes identifier, it is cast to float.
            - `name` is required for scalar, time_series and field features.
            - The order of the details must be respected. One cannot specify a detail in the feature_string_identifier string without specified the previous ones.
        """
        splitted_identifier = feature_string_identifier.split("::")

        feature_type = splitted_identifier[0]
        feature_details = [
            detail for detail in splitted_identifier[1].split("/") if detail
        ]

        assert feature_type in AUTHORIZED_FEATURE_TYPES, "feature_type not known"

        arg_names = AUTHORIZED_FEATURE_INFOS[feature_type]

        if feature_type == "scalar":
            return self.get_scalar(feature_details[0])
        elif feature_type == "time_series":
            return self.get_time_series(feature_details[0])
        elif feature_type == "field":
            kwargs = {arg_names[i]: detail for i, detail in enumerate(feature_details)}
            if "time" in kwargs:
                kwargs["time"] = float(kwargs["time"])
            return self.get_field(**kwargs)
        elif feature_type == "nodes":
            kwargs = {arg_names[i]: detail for i, detail in enumerate(feature_details)}
            if "time" in kwargs:
                kwargs["time"] = float(kwargs["time"])
            return self.get_nodes(**kwargs).flatten()

    def get_feature_from_identifier(
        self, feature_identifier: FeatureIdentifier
    ) -> FeatureType:
        """Retrieve a feature object based on a structured identifier dictionary.

        The `feature_identifier` must include a `"type"` key specifying the feature kind:
            - `"scalar"`       → calls `get_scalar(name)`
            - `"time_series"`  → calls `get_time_series(name)`
            - `"field"`        → calls `get_field(name, base_name, zone_name, location, time)`
            - `"nodes"`        → calls `get_nodes(base_name, zone_name, time)`

        Required keys:
            - `"type"`: one of `"scalar"`, `"time_series"`, `"field"`, or `"nodes"`
            - `"name"`: required for all types except `"nodes"`

        Optional keys depending on type:
            - `"base_name"`, `"zone_name"`, `"location"`, `"time"`: used in `"field"` and `"nodes"`

        Any omitted optional keys will rely on the default values mechanics of the class.

        Args:
            feature_identifier ( dict[str:Union[str, float]]):
                A dictionary encoding the feature type and its relevant parameters.

        Returns:
            FeatureType: The corresponding feature instance retrieved via the appropriate accessor.
        """
        feature_type, feature_details = get_feature_type_and_details_from(
            feature_identifier
        )

        if feature_type == "scalar":
            return self.get_scalar(**feature_details)
        elif feature_type == "time_series":
            return self.get_time_series(**feature_details)
        elif feature_type == "field":
            return self.get_field(**feature_details)
        elif feature_type == "nodes":
            return self.get_nodes(**feature_details).flatten()

    def get_features_from_identifiers(
        self, feature_identifiers: list[FeatureIdentifier]
    ) -> list[FeatureType]:
        """Retrieve features based on a list of structured identifier dictionaries.

        Elements of `feature_identifiers` must include a `"type"` key specifying the feature kind:
            - `"scalar"`       → calls `get_scalar(name)`
            - `"time_series"`  → calls `get_time_series(name)`
            - `"field"`        → calls `get_field(name, base_name, zone_name, location, time)`
            - `"nodes"`        → calls `get_nodes(base_name, zone_name, time)`

        Required keys:
            - `"type"`: one of `"scalar"`, `"time_series"`, `"field"`, or `"nodes"`
            - `"name"`: required for all types except `"nodes"`

        Optional keys depending on type:
            - `"base_name"`, `"zone_name"`, `"location"`, `"time"`: used in `"field"` and `"nodes"`

        Any omitted optional keys will rely on the default values mechanics of the class.

        Args:
            feature_identifiers (list[FeatureIdentifier]):
                A dictionary encoding the feature type and its relevant parameters.

        Returns:
            list[FeatureType]: List of corresponding feature instance retrieved via the appropriate accessor.
        """
        all_features_info = [
            get_feature_type_and_details_from(feat_id)
            for feat_id in feature_identifiers
        ]

        features = []
        for feature_type, feature_details in all_features_info:
            if feature_type == "scalar":
                features.append(self.get_scalar(**feature_details))
            elif feature_type == "time_series":
                features.append(self.get_time_series(**feature_details))
            elif feature_type == "field":
                features.append(self.get_field(**feature_details))
            elif feature_type == "nodes":
                features.append(self.get_nodes(**feature_details).flatten())
        return features

    def _add_feature(
        self,
        feature_identifier: FeatureIdentifier,
        feature: FeatureType,
    ) -> Self:
        """Add a feature to current sample.

        This method applies updates to scalars, time series, fields, or nodes
        using feature identifiers, and corresponding feature data.

        Args:
            feature_identifier (dict): A feature identifier.
            feature (FeatureType): A feature corresponding to the identifiers.

        Returns:
            Self: The updated sample

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """
        feature_type, feature_details = get_feature_type_and_details_from(
            feature_identifier
        )

        if feature_type == "scalar":
            if safe_len(feature) == 1:
                feature = feature[0]
            self.add_scalar(**feature_details, value=feature)
        elif feature_type == "time_series":
            self.add_time_series(
                **feature_details, time_sequence=feature[0], values=feature[1]
            )
        elif feature_type == "field":
            self.add_field(**feature_details, field=feature, warning_overwrite=False)
        elif feature_type == "nodes":
            physical_dim_arg = {
                k: v for k, v in feature_details.items() if k in ["base_name", "time"]
            }
            phys_dim = self.get_physical_dim(**physical_dim_arg)
            self.set_nodes(**feature_details, nodes=feature.reshape((-1, phys_dim)))

        return self

    def update_features_from_identifier(
        self,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
        features: Union[FeatureType, list[FeatureType]],
        in_place: bool = False,
    ) -> Self:
        """Update one or several features of the sample by their identifier(s).

        This method applies updates to scalars, time series, fields, or nodes
        using feature identifiers, and corresponding feature data. When `in_place=False`, a deep copy of the sample is created
        before applying updates, ensuring full isolation from the original.

        Args:
            feature_identifiers (dict or list of dict): One or more feature identifiers.
            features (FeatureType or list of FeatureType): One or more features corresponding
                to the identifiers.
            in_place (bool, optional): If True, modifies the current sample in place.
                If False, returns a deep copy with updated features.

        Returns:
            Self: The updated sample (either the current instance or a new copy).

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """
        assert isinstance(feature_identifiers, dict) or (
            isinstance(feature_identifiers, Iterable) and isinstance(features, Iterable)
        ), "Check types of feature_identifiers and features arguments"
        if isinstance(feature_identifiers, dict):
            feature_identifiers = [feature_identifiers]
            features = [features]

        sample = self if in_place else self.copy()

        for feat_id, feat in zip(feature_identifiers, features):
            sample._add_feature(feat_id, feat)

        return sample

    def from_features_identifier(
        self,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
    ) -> Self:
        """Extract features of the sample by their identifier(s) and return a new sample containing these features.

        This method applies updates to scalars, time series, fields, or nodes
        using feature identifiers

        Args:
            feature_identifiers (dict or list of dict): One or more feature identifiers.

        Returns:
            Self: New sample containing the provided feature identifiers

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """
        assert isinstance(feature_identifiers, dict) or isinstance(
            feature_identifiers, list
        ), "Check types of feature_identifiers argument"
        if isinstance(feature_identifiers, dict):
            feature_identifiers = [feature_identifiers]

        feature_types = set([feat_id["type"] for feat_id in feature_identifiers])

        # if field or node features are to extract, copy the source sample and delete all fields
        if "field" in feature_types or "nodes" in feature_types:
            source_sample = self.copy()
            source_sample.del_all_fields()

        sample = Sample()

        for feat_id in feature_identifiers:
            feature = self.get_feature_from_identifier(feat_id)

            # if trying to add a field or nodes, must check if the corresponding tree exists, and add it if not
            if feat_id["type"] in ["field", "nodes"]:
                # get time of current feature
                time = self.get_time_assignment(time=feat_id.get("time"))

                # if the constructed sample does not have a tree, add the one from the source sample, with no field
                if not sample.get_mesh(time):
                    sample.add_tree(source_sample.get_mesh(time))

            sample._add_feature(feat_id, feature)

        sample._extra_data = copy.deepcopy(self._extra_data)

        return sample

    def merge_features(self, sample: Self, in_place: bool = False) -> Self:
        """Merge features from another sample into the current sample.

        This method applies updates to scalars, time series, fields, or nodes
        using features from another sample. When `in_place=False`, a deep copy of the sample is created
        before applying updates, ensuring full isolation from the original.

        Args:
            sample (Sample): The sample from which features will be merged.
            in_place (bool, optional): If True, modifies the current sample in place.
                If False, returns a deep copy with updated features.

        Returns:
            Self: The updated sample (either the current instance or a new copy).
        """
        merged_dataset = self if in_place else self.copy()

        all_features_identifiers = sample.get_all_features_identifiers()
        all_features = sample.get_features_from_identifiers(all_features_identifiers)

        feature_types = set([feat_id["type"] for feat_id in all_features_identifiers])

        # if field or node features are to extract, copy the source sample and delete all fields
        if "field" in feature_types or "nodes" in feature_types:
            source_sample = sample.copy()
            source_sample.del_all_fields()

        for feat_id in all_features_identifiers:
            # if trying to add a field or nodes, must check if the corresponding tree exists, and add it if not
            if feat_id["type"] in ["field", "nodes"]:
                # get time of current feature
                time = sample.get_time_assignment(time=feat_id.get("time"))

                # if the constructed sample does not have a tree, add the one from the source sample, with no field
                if not merged_dataset.get_mesh(time):
                    merged_dataset.add_tree(source_sample.get_mesh(time))

        return merged_dataset.update_features_from_identifier(
            feature_identifiers=all_features_identifiers,
            features=all_features,
            in_place=in_place,
        )

    # -------------------------------------------------------------------------#
    def save(self, dir_path: Union[str, Path], overwrite: bool = False) -> None:
        """Save the Sample in directory `dir_path`.

        Args:
            dir_path (Union[str,Path]): relative or absolute directory path.
            overwrite (bool): target directory overwritten if True.
        """
        dir_path = Path(dir_path)

        if dir_path.is_dir():
            if overwrite:
                shutil.rmtree(dir_path)
                logger.warning(f"Existing {dir_path} directory has been reset.")
            elif len(list(dir_path.glob("*"))):
                raise ValueError(
                    f"directory {dir_path} already exists and is not empty. Set `overwrite` to True if needed."
                )

        dir_path.mkdir(exist_ok=True)

        mesh_dir = dir_path / "meshes"

        if self._meshes is not None:
            mesh_dir.mkdir()
            for i, time in enumerate(self._meshes.keys()):
                outfname = mesh_dir / f"mesh_{i:09d}.cgns"
                status = CGM.save(
                    str(outfname), self._meshes[time], links=self._links[time]
                )
                logger.debug(f"save -> {status=}")

        scalars_names = self.get_scalar_names()
        if len(scalars_names) > 0:
            scalars = []
            for s_name in scalars_names:
                scalars.append(self.get_scalar(s_name))
            scalars = np.array(scalars).reshape((1, -1))
            header = ",".join(scalars_names)
            np.savetxt(
                dir_path / "scalars.csv",
                scalars,
                header=header,
                delimiter=",",
                comments="",
            )

        time_series_names = self.get_time_series_names()
        if len(time_series_names) > 0:
            for ts_name in time_series_names:
                ts = self.get_time_series(ts_name)
                data = np.vstack((ts[0], ts[1])).T
                header = ",".join(["t", ts_name])
                np.savetxt(
                    dir_path / f"time_series_{ts_name}.csv",
                    data,
                    header=header,
                    delimiter=",",
                    comments="",
                )

    @classmethod
    def load_from_dir(cls, dir_path: Union[str, Path]) -> Self:
        """Load the Sample from directory `dir_path`.

        This is a class method, you don't need to instantiate a `Sample` first.

        Args:
            dir_path (Union[str,Path]): Relative or absolute directory path.

        Returns:
            Sample

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample = Sample.load_from_dir(dir_path)
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)

        Note:
            It calls 'load' function during execution.
        """
        dir_path = Path(dir_path)
        instance = cls()
        instance.load(dir_path)
        return instance

    def load(self, dir_path: Union[str, Path]) -> None:
        """Load the Sample from directory `dir_path`.

        Args:
            dir_path (Union[str,Path]): Relative or absolute directory path.

        Raises:
            FileNotFoundError: Triggered if the provided directory does not exist.
            FileExistsError: Triggered if the provided path is a file instead of a directory.

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample = Sample()
                sample.load(dir_path)
                print(sample)
                >>> Sample(3 scalars, 1 timestamp, 3 fields)

        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f'Directory "{dir_path}" does not exist. Abort')

        if not dir_path.is_dir():
            raise FileExistsError(f'"{dir_path}" is not a directory. Abort')

        meshes_dir = dir_path / "meshes"
        if meshes_dir.is_dir():
            meshes_names = list(meshes_dir.glob("*"))
            nb_meshes = len(meshes_names)
            self._meshes = {}
            self._links = {}
            self._paths = {}
            for i in range(nb_meshes):
                tree, links, paths = CGM.load(str(meshes_dir / f"mesh_{i:09d}.cgns"))
                time = CGH.get_time_values(tree)

                self._meshes[time], self._links[time], self._paths[time] = (
                    tree,
                    links,
                    paths,
                )
                for i in range(len(self._links[time])):  # pragma: no cover
                    self._links[time][i][0] = str(meshes_dir / self._links[time][i][0])

        scalars_fname = dir_path / "scalars.csv"
        if scalars_fname.is_file():
            names = np.loadtxt(
                scalars_fname, dtype=str, max_rows=1, delimiter=","
            ).reshape((-1,))
            scalars = np.loadtxt(
                scalars_fname, dtype=float, skiprows=1, delimiter=","
            ).reshape((-1,))
            for name, value in zip(names, scalars):
                self.add_scalar(name, value)

        time_series_files = list(dir_path.glob("time_series_*.csv"))
        for ts_fname in time_series_files:
            names = np.loadtxt(ts_fname, dtype=str, max_rows=1, delimiter=",").reshape(
                (-1,)
            )
            assert names[0] == "t"
            times_and_val = np.loadtxt(ts_fname, dtype=float, skiprows=1, delimiter=",")
            self.add_time_series(names[1], times_and_val[:, 0], times_and_val[:, 1])

    # # -------------------------------------------------------------------------#
    def __str__(self) -> str:
        """Return a string representation of the sample.

        Returns:
            str: A string representation of the overview of sample content.
        """
        # TODO rewrite using self.get_all_features_identifiers()
        str_repr = "Sample("

        # scalars
        nb_scalars = len(self.get_scalar_names())
        str_repr += f"{nb_scalars} scalar{'' if nb_scalars == 1 else 's'}, "

        # time series
        nb_ts = len(self.get_time_series_names())
        str_repr += f"{nb_ts} time series, "

        # fields
        times = self.get_all_mesh_times()
        nb_timestamps = len(times)
        str_repr += f"{nb_timestamps} timestamp{'' if nb_timestamps == 1 else 's'}, "

        field_names = set()
        for time in times:
            ## Need to include all possible location within the count
            base_names = self.get_base_names(time=time)
            for bn in base_names:
                zone_names = self.get_zone_names(base_name=bn)
                for zn in zone_names:
                    field_names = field_names.union(
                        self.get_field_names(zone_name=zn, time=time, location="Vertex")
                        + self.get_field_names(
                            zone_name=zn, time=time, location="EdgeCenter"
                        )
                        + self.get_field_names(
                            zone_name=zn, time=time, location="FaceCenter"
                        )
                        + self.get_field_names(
                            zone_name=zn, time=time, location="CellCenter"
                        )
                    )
        nb_fields = len(field_names)
        str_repr += f"{nb_fields} field{'' if nb_fields == 1 else 's'}, "

        # CGNS tree
        if self._meshes is None:
            str_repr += "no tree, "
        else:
            # TODO
            pass

        if str_repr[-2:] == ", ":
            str_repr = str_repr[:-2]
        str_repr = str_repr + ")"

        return str_repr

    @model_serializer()
    def serialize_model(self):
        """Serialize the model to a dictionary."""
        return {
            "mesh_base_name": self._mesh_base_name,
            "mesh_zone_name": self._mesh_zone_name,
            "meshes": self._meshes,
            "scalars": self._scalars,
            "time_series": self._time_series,
            "links": self._links,
            "paths": self._paths,
        }
