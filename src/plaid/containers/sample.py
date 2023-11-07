# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

try: # pragma: no cover
    from typing import Self
except ImportError: # pragma: no cover
    from typing import Any as Self

import glob
import logging
import os
from typing import Optional, Union

import CGNS.MAP as CGM
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import numpy as np
from CGNS.PAT.cgnsutils import __CHILDREN__
from CGNS.PAT.cgnsutils import __LABEL__ as __TYPE__
from CGNS.PAT.cgnsutils import __NAME__
from CGNS.PAT.cgnsutils import __VALUE__ as __DATA__

from plaid.utils import cgns_helper as CGH

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s',
    level=logging.INFO)

# %% Globals

CGNS_element_names = [
    "ElementTypeNull",
    "ElementTypeUserDefined",
    "NODE",
    "BAR_2",
    "BAR_3",
    "TRI_3",
    "TRI_6",
    "QUAD_4",
    "QUAD_8",
    "QUAD_9",
    "TETRA_4",
    "TETRA_10",
    "PYRA_5",
    "PYRA_14",
    "PENTA_6",
    "PENTA_15",
    "PENTA_18",
    "HEXA_8",
    "HEXA_20",
    "HEXA_27",
    "MIXED",
    "PYRA_13",
    "NGON_n",
    "NFACE_n",
    "BAR_4",
    "TRI_9",
    "TRI_10",
    "QUAD_12",
    "QUAD_16",
    "TETRA_16",
    "TETRA_20",
    "PYRA_21",
    "PYRA_29",
    "PYRA_30",
    "PENTA_24",
    "PENTA_38",
    "PENTA_40",
    "HEXA_32",
    "HEXA_56",
    "HEXA_64"]
"""List of element type names commonly used in Computational Fluid Dynamics (CFD). These names represent different types of finite elements that are used to discretize physical domains for numerical analysis."""
# %% Functions


def show_cgns_tree(tree, offset=''):
    """Recursively prints the CGNS Tree structure. It displays the hierarchy of nodes and branches in a CGNS tree.

    Args:
        tree (list): The CGNS tree structure to be printed.
        offset (str, optional): The character used for indentation between tree branches. Defaults to an empty string ('').
    """
    if not (isinstance(tree, list)):
        if tree is None: # pragma: no cover
            return True
        else:
            raise TypeError(f"{type(tree)=}, but should be a list or None")
    assert (len(tree) == 4)
    if isinstance(tree[__DATA__], np.ndarray):
        if '|S' in str(tree[__DATA__].dtype):
            data = tree[__DATA__].tobytes()
        elif tree[__DATA__].size < 10:
            data = tree[__DATA__]
        else:
            data = f"[{tree[__DATA__].dtype};{tree[__DATA__].shape}]"
    else:
        data = tree[__DATA__]
    print(
        offset +
        f"""- "{tree[__NAME__]}"({tree[__TYPE__]}), {len(tree[__CHILDREN__])} children, data({type(tree[__DATA__])}): {data}""")
    for stree in tree[__CHILDREN__]:
        show_cgns_tree(stree, offset=offset + '    ')

# %% Classes


CGNSTree = list
"""A CGNSTree is a list
"""
CGNSNode = list
"""A CGNSNode is a list
"""
LinkType = list[str]
"""A link is a list containing 4 str [target_dir_path,target_file_name,target_node_name,local_node_name]

        - target_dir_path (optional)
        - target_file_name
        - target_node_name: absolute tree path
        - local_node_name: absolute tree path

    See https://chlone.sourceforge.net/sids-to-python.html#links
"""
PathType = tuple
"""A PathType is a tuple
"""
ScalarType = Union[float, int]
"""A ScalarType is an Union[float,int]
"""
FieldType = np.ndarray
"""A FieldType is a np.ndarray
"""
TimeSequenceType = np.ndarray
"""A TimeSequenceType is a np.ndarray
"""
TimeSeriesType = tuple[TimeSequenceType, FieldType]
"""A TimeSeriesType is a tuple[TimeSequenceType,FieldType]
"""


class Sample(object):
    """Represents a single sample. It contains data and information related to a single observation or measurement within a dataset.
    """

    def __init__(self, directory_path: str = None,
                 mesh_base_name: str = 'Base') -> None:
        """Initialize an empty :class:`Sample <plaid.containers.sample.Sample>`.

        Args:
            directory_path (str, optional): The path from which to load PLAID sample files.
            mesh_base_name (str, optional): The base name for the mesh. Defaults to 'Base'.

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
        self._meshes: dict[float, CGNSTree] = None
        self._scalars: dict[str, ScalarType] = None
        self._time_series: dict[str, TimeSeriesType] = None

        self._links: dict[float, list[LinkType]] = None
        self._paths: dict[float, list[PathType]] = None

        self._mesh_base_name: str = mesh_base_name

        if directory_path is not None:
            self.load(directory_path)

        self._defaults: dict = {
            "active_base": None,
            "active_zone": None,
            "active_time": None
        }

    # -------------------------------------------------------------------------#
    def set_default_base(self, base_name: str, time: float = None) -> None:
        """Set the default base for the specified time (that will also be set as default if provided).
        The default base is a reference point for various operations in the system.

        Args:
            base_name (str): The name of the base to be set as the default.
            time (float, optional): The time at which the base should be set as default. If not provided, the default base and active zone will be set with the default time.

        Raises:
            ValueError: If the specified base does not exist at the given time.

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)

                # Set "BaseA" as the default base for the default time
                sample.set_default_base("BaseA")

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

    def set_default_base_zone(self, base_name: str, zone_name: str, time: float = None) -> None:
        """Set the default base and active zone for the specified time (that will also be set as default if provided).
        The default base and active zone serve as reference points for various operations in the system.

        Args:
            base_name (str): The name of the base to be set as the default.
            zone_name (str): The name of the zone to be set as the active zone.
            time (float, optional): The time at which the base and zone should be set as default. If not provided, the default base and active zone will be set with the default time.

        Raises:
            ValueError: If the specified base or zone does not exist at the given time

        Example:
            .. code-block:: python

                from plaid.containers.sample import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)

                # Set "BaseA" as the default base and "ZoneX" as the active zone for the default time
                sample.default_base_zone("BaseA", "ZoneX")

                # Set "BaseB" as the default base and "ZoneY" as the active zone for a specific time
                sample.default_base_zone("BaseB", "ZoneY", 0.5)

                # You can now use class functions with "BaseB" as default base with "ZoneY" as default zone and 0.5 as default time
                print(sample.get_zone_type()) # type of the zone "ZoneY" of base "BaseB" at 0.5
                >>> Unstructured
        """
        self.set_default_base(base_name, time)
        if zone_name in (self._defaults["active_zone"], None):
            return
        if not self.has_zone(zone_name, base_name, time):
            raise ValueError(f"zone {zone_name} does not exist for the base {base_name} at time {time}")

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

                 # Set the default time to 0.5 seconds
                sample.set_default_time(0.5)

                # You can now use class functions with 0.5 as default time
                print(sample.show_tree()) # show the cgns tree at the time 0.5
                >>> ...
        """
        if time in (self._defaults["active_time"], None):
            return
        if not time in self.get_all_mesh_times():
            raise ValueError(f"time {time} does not exist in mesh times")

        self._defaults["active_time"] = time

    def get_time_assignment(self, time: float = None) -> float:
        """Retrieve the default time for the CGNS operations.
        If there are available time steps, it will return the first one; otherwise, it will return 0.0.

        Args:
            base_name (str, optional): The time value provided for the operation. Defaults to None.

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

    def get_base_assignment(self, base_name: str = None) -> str:
        """Retrieve the default base name for the CGNS operations.
        This function calculates the attributed base for a specific operation based on the
        default base set in the system.

        Args:
            base_name (str, optional): The name of the base to attribute the operation to. If not provided, the default base set in the system will be used.

        Raises:
            KeyError: If no default base can be determined based on the provided or default value.

        Returns:
            str: The attributed base name.

        Note:
            If no specific base name is provided, the function will use the default base provided to the system.
        """
        base_name = base_name or self._defaults.get("active_base")

        if base_name:
            return base_name

        base_names = self.get_base_names()
        if len(base_names) == 0:
            return None
        elif len(base_names) == 1:
            # logging.info(f"No default base provided. Taking the only base available: {base_names[0]}")
            return base_names[0]

        raise KeyError(f"No default base provided among {base_names}")

    def get_zone_assignment(self, zone_name: str = None, base_name: str = None) -> str:
        """Retrieve the default zone name for the CGNS operations.
        This function calculates the attributed zone for a specific operation based on the
        default zone set in the system, within the specified base.

        Args:
            zone_name (str, optional): The name of the zone to attribute the operation to. If not provided, the default zone set in the system within the specified base will be used.
            base_name (str, optional): The name of the base within which the zone should be attributed. If not provided, the default base set in the system will be used.

        Raises:
            KeyError: If no default zone can be determined based on the provided or default values.

        Returns:
            str: The attributed zone name.

        Note:
            If neither a specific zone name nor a specific base name is provided, the function will use the default zone provided to the system.
        """
        zone_name = zone_name or self._defaults.get("active_zone")

        if zone_name:
            return zone_name

        base_name = self.get_base_assignment(base_name)
        zone_names = self.get_zone_names(base_name)
        if len(zone_names) == 0:
            return None
        elif len(zone_names) == 1:
            # logging.info(f"No default zone provided. Taking the only zone available: {zone_names[0]} in default base: {base_name}")
            return zone_names[0]

        raise KeyError(f"No default zone provided among {zone_names} in the default base: {base_name}")

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
            show_cgns_tree(self._meshes[time])

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

    def get_mesh(self, time: float = None) -> CGNSTree:
        """Retrieve the CGNS tree structure for a specified time step, if available.

        Args:
            time (float, optional): The time step for which to retrieve the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSTree: The CGNS tree structure for the specified time step if available; otherwise, returns None.
        """
        time = self.get_time_assignment(time)
        return self._meshes[time] if (self._meshes is not None) else None

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
        else:
            raise KeyError(
                "meshes is already set, you cannot overwrite it, delete it first or extend it with `Sample.add_tree`")

    def add_tree(self, tree: CGNSTree, time: float = None) -> CGNSTree:
        """Merge a CGNS tree to the already existing tree.

        Args:
            tree (CGNSTree): The CGNS tree to be merged. If a Base node already exists, it is ignored.
            time (float, optional): The time step for which to add the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSTree: The merged CGNS tree.
        """
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
            base_nodes = CGU.getNodesFromTypeSet(tree, 'CGNSBase_t')
            for _, node in base_nodes:
                if (node[__NAME__] not in local_bases):
                    self._meshes[time][__CHILDREN__].append(node)
                else:
                    logger.warning(
                        f"base <{node[__NAME__]}> already exists in self._tree --> ignored")

        base_names = self.get_base_names(time=time)
        for base_name in base_names:
            base_node = self.get_base(base_name, time=time)
            if CGU.getValueByPath(base_node, "Time/TimeValues") is None:
                baseIterativeData_node = CGL.newBaseIterativeData(
                    base_node, 'Time', 1)
                TimeValues_node = CGU.newNode(
                    'TimeValues', None, [], CGK.DataArray_ts, baseIterativeData_node)
                CGU.setValue(TimeValues_node, np.array([time]))

        return self._meshes[time]

    """def link_(self, sample:Sample, sample_index:int, time_index:int) -> CGNSTree:
        #NOT IMPLEMENTED YET
        #see https://pycgns.github.io/MAP/sids-to-python.html#links
        #difficulty is to link only the geometrical objects, which can be complex

        #https://pycgns.github.io/MAP/examples.html#save-with-links
        #When you load a file all the linked-to files areresolved to produce a full CGNS/Python tree with actual node data.

        tree = sample.get_tree(time)
        self.add_tree(tree, time)

        time_index = list[sample._meshes.keys()].index(time)

        if self._links[time] == None:
            self._links[time] = []

        self._links[time].append([])

        folder = "../../sample_"+str(sample_index).zfill(9)+os.sep+"mesh_"+str(time_index).zfill(9)+".cgns"
        return tree"""

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
        if base_node is None:
            raise ValueError(f"there is no base called {base_name} at the time {time} in this sample")

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
        if base_node is None:
            raise ValueError(f"there is no base called {base_name} at the time {time} in this sample")

        return base_node[1][1]

    def init_base(self, topological_dim: int, physical_dim: int,
                  base_name: str = None, time: float = None) -> CGNSNode:
        """Create a Base node named `base_name` if it doesn't already exists.

        Args:
            topological_dim (int): Cell dimension, see [CGNS standard](https://pycgns.github.io/PAT/lib.html#CGNS.PAT.cgnslib.newCGNSBase).
            physical_dim (int): Ambient space dimension, see [CGNS standard](https://pycgns.github.io/PAT/lib.html#CGNS.PAT.cgnslib.newCGNSBase).
            base_name (str): If not specified, uses `mesh_base_name` specified in Sample initialization. Defaults to None.
            time (float, optional): The time at which to initialize the base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSNode: The created Base node.
        """
        time = self.get_time_assignment(time)

        if base_name is None:
            base_name = self._mesh_base_name + "_" + \
                str(topological_dim) + "_" + str(physical_dim)

        self.init_tree(time)
        if not (self.has_base(base_name)):
            base_node = CGL.newCGNSBase(
                self._meshes[time],
                base_name,
                topological_dim,
                physical_dim)

        base_names = self.get_base_names(time=time)
        for base_name in base_names:
            base_node = self.get_base(base_name, time=time)
            if CGU.getValueByPath(base_node, "Time/TimeValues") is None:
                baseIterativeData_node = CGL.newBaseIterativeData(
                    base_node, 'Time', 1)
                TimeValues_node = CGU.newNode(
                    'TimeValues', None, [], CGK.DataArray_ts, baseIterativeData_node)
                CGU.setValue(TimeValues_node, np.array([time]))

        return base_node

    def get_base_names(self, full_path: bool = False,
                       unique: bool = False, time: float = None) -> list[str]:
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
                    self._meshes[time], full_path, unique)
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
        return (base_name in self.get_base_names(time=time))

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
        base_name = self.get_base_assignment(base_name)

        if (self._meshes is None) or (self._meshes[time] is None):
            return None

        return CGU.getNodeByPath(self._meshes[time], f'/CGNSTree/{base_name}')

    # -------------------------------------------------------------------------#
    def init_zone(self, zone_shape: np.ndarray, zone_type: str = CGK.Unstructured_s,
                  zone_name: str = None, base_name: str = None, time: float = None) -> CGNSNode:
        """Initialize a new zone within a CGNS base.

        Args:
            zone_shape (np.ndarray): An array specifying the shape or dimensions of the zone.
            zone_type (str, optional): The type of the zone. Defaults to CGK.Unstructured_s.
            zone_name (str, optional): The name of the zone to initialize.
            base_name (str, optional): The name of the base to which the zone will be added. If not provided, the zone will be added to the currently active base. Defaults to None.
            time (float, optional): The time at which to initialize the zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Raises:
            KeyError: If the specified base does not exist. You can create a base using `Sample.init_base(base_name)`.

        Returns:
            CGLNode: The newly initialized zone node within the CGNS tree.
        """
        # init_tree will look for default time
        self.init_tree(time)
        # get_base will look for default base_name and time
        base_node = self.get_base(base_name, time)
        if base_node is None:
            raise KeyError(
                f"there is no base <{base_name}>, you should first create one with `Sample.init_base({base_name=})`")
        zone_name = self.get_zone_assignment(zone_name, base_name)
        zone_node = CGL.newZone(base_node, zone_name, zone_shape, zone_type)
        return zone_node

    def get_zone_names(self, base_name: str = None, full_path: bool = False,
                       unique: bool = False, time: float = None) -> list[str]:
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
            z_paths = CGU.getPathsByTypeSet(base_node, 'CGNSZone_t')
            for pth in z_paths:
                s_pth = pth.split('/')
                assert (len(s_pth) == 2)
                assert (s_pth[0] == base_name or base_name is None)
                if full_path:
                    zone_paths.append(pth)
                else:
                    zone_paths.append(s_pth[1])

        if unique:
            return list(set(zone_paths))
        else:
            return zone_paths

    def has_zone(self, zone_name: str, base_name: str = None,
                 time: float = None) -> bool:
        """Check if the CGNS tree contains a Zone with the specified name within a specific Base and time.

        Args:
            zone_name (str): The name of the Zone to check for.
            base_name (str, optional): The name of the Base where the Zone should be located. If not provided, the function checks all bases. Defaults to None.
            time (float, optional): The time at which to check for the Zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            bool: `True` if the CGNS tree has a Zone called `zone_name` in a Base called `base_name`, else return `False`.
        """
        # get_zone_names will look for default base_name and time
        return (zone_name in self.get_zone_names(base_name, time=time))

    def get_zone(self, zone_name: str = None, base_name: str = None,
                 time: float = None) -> CGNSNode:
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
            return None

        # _zone_attribution will look for default base_name
        zone_name = self.get_zone_assignment(zone_name, base_name)
        if zone_name is None:
            return None

        return CGU.getNodeByPath(base_node, zone_name)

    def get_zone_type(self, zone_name: str = None,
                      base_name: str = None, time: float = None) -> str:
        """ Get the type of a specific zone at a specified time.

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
                f"there is no base/zone <{base_name}/{zone_name}>, you should first create one with `Sample.init_zone({zone_name=},{base_name=})`")
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
        if self._scalars is None:
            self._scalars = {name: value}
        else:
            self._scalars[name] = value

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
            self, name: str, time_sequence: TimeSequenceType, values: FieldType) -> None:
        """Add a time series to the sample (Sample).

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
        assert (len(time_sequence) == len(values)
                ), "time sequence and values do not have the same size"
        if self._time_series is None:
            self._time_series = {name: (time_sequence, values)}
        else:
            self._time_series[name] = (time_sequence, values)

    # -------------------------------------------------------------------------#
    def get_nodes(self, zone_name: str = None, base_name: str = None,
                  time: float = None) -> Optional[np.ndarray]:
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

        grid_paths = CGU.getAllNodesByTypeSet(
            search_node, ['GridCoordinates_t'])
        if len(grid_paths) == 1:
            grid_node = CGU.getNodeByPath(search_node, grid_paths[0])
            array_x = CGU.getValueByPath(
                grid_node, 'GridCoordinates/CoordinateX')
            array_y = CGU.getValueByPath(
                grid_node, 'GridCoordinates/CoordinateY')
            array_z = CGU.getValueByPath(
                grid_node, 'GridCoordinates/CoordinateZ')
            if array_z is None:
                array = np.concatenate(
                    (array_x.reshape((-1, 1)), array_y.reshape((-1, 1))), axis=1)
            else:
                array = np.concatenate((array_x.reshape(
                    (-1, 1)), array_y.reshape((-1, 1)), array_z.reshape((-1, 1))), axis=1)
            return array
        elif len(grid_paths) > 1:
            raise TypeError(
                f"Found {len(grid_paths)} <GridCoordinates> nodes, should find only one")

    get_points = get_nodes
    get_vertices = get_nodes

    def set_nodes(self, nodes: np.ndarray, zone_name: str = None,
                  base_name: str = None, time: float = None) -> None:
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
                f"there is no base/zone <{base_name}/{zone_name}>, you should first create one with `Sample.init_zone({zone_name=},{base_name=})`")

        coord_type = [CGK.CoordinateX_s, CGK.CoordinateY_s, CGK.CoordinateZ_s]
        for i_dim in range(nodes.shape[1]):
            CGL.newCoordinates(
                zone_node, coord_type[i_dim], np.asfortranarray(nodes[:, i_dim]))

    set_points = set_nodes
    set_vertices = set_nodes

    # -------------------------------------------------------------------------#
    def get_elements(self, zone_name: str = None, base_name: str = None,
                     time: float = None) -> dict[str, np.ndarray]:
        """Retrieve element connectivity data for a specified zone, base, and time.

        Args:
            zone_name (str, optional): The name of the zone for which element connectivity data is requested. Defaults to None, indicating the default zone.
            base_name (str, optional): The name of the base for which element connectivity data is requested. Defaults to None, indicating the default base.
            time (float, optional): The time at which element connectivity data is requested. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            dict[str,np.ndarray]: A dictionary where keys are element type namesand values are NumPy arrays representing the element connectivity data.
            The NumPy arrays have shape (num_elements, num_nodes_per_element), and element indices are 0-based.
        """
        # get_zone will look for default base_name, zone_name and time
        zone_node = self.get_zone(zone_name, base_name, time)

        if zone_node is None:
            return {}

        elements = {}
        elem_paths = CGU.getAllNodesByTypeSet(zone_node, ['Elements_t'])

        for elem in elem_paths:
            elem_node = CGU.getNodeByPath(zone_node, elem)
            val = CGU.getValue(elem_node)
            elem_type = CGNS_element_names[val[0]]
            elem_size = int(elem_type.split('_')[-1])
            elem_range = CGU.getValueByPath(
                elem_node, 'ElementRange')  # TODO elem_range is unused
            # -1 is to get back indexes starting at 0
            elements[elem_type] = CGU.getValueByPath(
                elem_node, 'ElementConnectivity').reshape(
                (-1, elem_size)) - 1

        return elements

    # -------------------------------------------------------------------------#
    def get_field_names(self, zone_name: str = None, base_name: str = None,
                        location: str = 'Vertex', time: float = None) -> set[str]:
        """Get a set of field names associated with a specified zone, base, location, and time.

        Args:
            zone_name (str, optional): The name of the zone to search for. Defaults to None.
            base_name (str, optional): The name of the base to search for. Defaults to None.
            location (str, optional): The desired grid location where the field is defined. Defaults to 'Vertex'.
            time (float, optional): The specific time at which to retrieve field names. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            set[str]: A set containing the names of the fields that match the specified criteria.
        """
        def get_field_names_one_base(base_name: str) -> list[str]:

            # get_zone will look for default zone_name, base_name, time
            search_node = self.get_zone(zone_name, base_name, time)

            if search_node is None:
                return []

            names = []
            solution_paths = CGU.getPathsByTypeSet(search_node, [CGK.FlowSolution_t])
            for f_path in solution_paths:
                if CGU.getValueByPath(
                        search_node, f_path + '/GridLocation').tobytes().decode() != location:
                    continue

                f_node = CGU.getNodeByPath(search_node, f_path)
                for path in CGU.getPathByTypeFilter(
                        f_node, CGK.DataArray_t):
                    field_name = path.split('/')[-1]
                    if not (field_name == 'GridLocation'):
                        names.append(field_name)

            return names

        if base_name is None:
            # get_base_names will look for default time
            base_names = self.get_base_names(time)
        else:
            base_names = [base_name]

        all_names = []
        for bn in base_names:
            all_names += get_field_names_one_base(bn)

        all_names.sort()
        all_names = list(set(all_names))

        return all_names

    def get_field(self, name: str, zone_name: str = None, base_name: str = None,
                  location: str = 'Vertex', time: float = None) -> FieldType:
        """Retrieve a field with a specified name from a given zone, base, location, and time.

        Args:
            name (str): The name of the field to retrieve.
            zone_name (str, optional): The name of the zone to search for. Defaults to None.
            base_name (str, optional): The name of the base to search for. Defaults to None.
            location (str, optional): The location at which to retrieve the field. Defaults to 'Vertex'.
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
            if CGU.getValueByPath(
                    search_node, f_path + '/GridLocation').tobytes().decode() == location:
                field = CGU.getValueByPath(search_node, f_path + '/' + name)

                if field is None:
                    field = np.empty((0,))
                else:
                    is_empty = False
                full_field.append(field)

        if is_empty:
            return None
        else:
            return np.concatenate(full_field)

    def add_field(self, name: str, field: FieldType, zone_name: str = None,
                  base_name: str = None, location: str = 'Vertex', time: float = None) -> None:
        """Add a field to a specified zone in the grid.

        Args:
            name (str): The name of the field to be added.
            field (FieldType): The field data to be added.
            zone_name (str, optional): The name of the zone where the field will be added. Defaults to None.
            base_name (str, optional): The name of the base where the zone is located. Defaults to None.
            location (str, optional): The grid location where the field will be stored. Defaults to 'Vertex'.
            time (float, optional): The time associated with the field. Defaults to 0.

        Raises:
            KeyError: Raised if the specified zone does not exist in the given base.
        """
        # init_tree will look for default time
        self.init_tree(time)
        # get_zone will look for default zone_name, base_name and time
        zone_node = self.get_zone(zone_name, base_name, time)

        if zone_node is None:
            raise KeyError(
                f"there is no Zone with name {zone_name} in base {base_name}. Did you check topological and physical dimensions ?")

        # solution_paths = CGU.getPathsByTypeOrNameList(self._tree, '/.*/.*/FlowSolution_t')
        solution_paths = CGU.getPathsByTypeSet(zone_node, 'FlowSolution_t')
        has_FlowSolution_with_location = False
        if len(solution_paths) > 0:
            for s_path in solution_paths:
                val_location = CGU.getValueByPath(
                    zone_node, f'{s_path}/GridLocation').tobytes().decode()
                if val_location == location:
                    has_FlowSolution_with_location = True

        if not (has_FlowSolution_with_location):
            CGL.newFlowSolution(
                zone_node,
                f'{location}Fields',
                gridlocation=location)

        solution_paths = CGU.getPathsByTypeSet(zone_node, 'FlowSolution_t')
        assert (len(solution_paths) > 0)

        for s_path in solution_paths:
            val_location = CGU.getValueByPath(
                zone_node, f'{s_path}/GridLocation').tobytes().decode()

            if val_location != location:
                continue

            field_node = CGU.getNodeByPath(zone_node, f'{s_path}/{name}')

            if field_node is None:
                flow_solution_node = CGU.getNodeByPath(zone_node, s_path)
                # CGL.newDataArray(flow_solution_node, name, np.asfortranarray(np.copy(field), dtype=np.float64))
                CGL.newDataArray(
                    flow_solution_node, name, np.asfortranarray(field))
                # res =  [name, np.asfortranarray(field, dtype=np.float32), [], 'DataArray_t']
                # print(field.shape)
                # flow_solution_node[2].append(res)
            else:
                logger.warning(
                    f"field node with name {name} already exists -> data will be replaced")
                CGU.setValue(field_node, np.asfortranarray(field))

    # -------------------------------------------------------------------------#
    def save(self, dir_path: str) -> None:
        """Save the Sample in directory `dir_path`.

        Args:
            dir_path (str): relative or absolute directory path.
        """
        if os.path.isdir(dir_path):
            if len(glob.glob(os.path.join(dir_path, '*'))):
                raise ValueError(
                    f"directory {dir_path} already exists and is not empty")
        else:
            os.makedirs(dir_path)

        mesh_dir = os.path.join(dir_path, "meshes")

        if self._meshes is not None:
            os.makedirs(mesh_dir)
            for i, time in enumerate(self._meshes.keys()):
                outfname = os.path.join(mesh_dir, f"mesh_{i:09d}.cgns")
                status = CGM.save(
                    outfname,
                    self._meshes[time],
                    links=self._links[time])
                logger.debug(f"save -> {status=}")

        scalars_names = self.get_scalar_names()
        if len(scalars_names) > 0:
            scalars = []
            for s_name in scalars_names:
                scalars.append(self.get_scalar(s_name))
            scalars = np.array(scalars).reshape((1, -1))
            header = ','.join(scalars_names)
            np.savetxt(
                os.path.join(
                    dir_path,
                    'scalars.csv'),
                scalars,
                header=header,
                delimiter=',',
                comments='')

        time_series_names = self.get_time_series_names()
        if len(time_series_names) > 0:
            for ts_name in time_series_names:
                ts = self.get_time_series(ts_name)
                data = np.vstack((ts[0], ts[1])).T
                header = ','.join(['t', ts_name])
                np.savetxt(
                    os.path.join(
                        dir_path,
                        f'time_series_{ts_name}.csv'),
                    data,
                    header=header,
                    delimiter=',',
                    comments='')

    @classmethod
    def load_from_dir(cls, dir_path: str) -> Self:
        """Load the Sample from directory `dir_path`.

        This is a class method, you don't need to instantiate a `Sample` first.

        Args:
            dir_path (str): Relative or absolute directory path.

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
        instance = cls()
        instance.load(dir_path)
        return instance

    def load(self, dir_path: str) -> None:
        """Load the Sample from directory `dir_path`.

        Args:
            dir_path (str): Relative or absolute directory path.

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
        if not os.path.exists(dir_path):
            raise FileNotFoundError(
                f"Directory \"{dir_path}\" does not exist. Abort")

        if not os.path.isdir(dir_path):
            raise FileExistsError(f"\"{dir_path}\" is not a directory. Abort")

        meshes_dir = os.path.join(dir_path, 'meshes')
        if os.path.isdir(meshes_dir):
            meshes_names = glob.glob(os.path.join(meshes_dir, '*'))
            nb_meshes = len(meshes_names)
            self._meshes = {}
            self._links = {}
            self._paths = {}
            for i in range(nb_meshes):
                tree, links, paths = CGM.load(
                    os.path.join(meshes_dir, f"mesh_{i:09d}.cgns"))
                time = CGH.get_time_values(tree)
                self._meshes[time], self._links[time], self._paths[time] = tree, links, paths

        scalars_fname = os.path.join(dir_path, 'scalars.csv')
        if os.path.isfile(scalars_fname):
            names = np.loadtxt(
                scalars_fname,
                dtype=str,
                max_rows=1,
                delimiter=',').reshape(
                (-1,
                 ))
            scalars = np.loadtxt(
                scalars_fname,
                dtype=float,
                skiprows=1,
                delimiter=',').reshape(
                (-1,
                 ))
            for name, value in zip(names, scalars):
                self.add_scalar(name, value)

        time_series_files = glob.glob(
            os.path.join(dir_path, 'time_series_*.csv'))
        for ts_fname in time_series_files:
            names = np.loadtxt(
                ts_fname, dtype=str, max_rows=1, delimiter=',').reshape(
                (-1,))
            assert names[0] == "t"
            times_and_val = np.loadtxt(
                ts_fname, dtype=float, skiprows=1, delimiter=',')
            self.add_time_series(
                names[1], times_and_val[:, 0], times_and_val[:, 1])

    # -------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """Return a string representation of the sample.

        Returns:
            str: A string representation of the overview of sample content.
        """
        str_repr = "Sample("

        # scalars
        nb_scalars = len(self.get_scalar_names())
        str_repr += f"{nb_scalars} scalar{'' if nb_scalars==1 else 's'}, "

        # fields
        times = self.get_all_mesh_times()
        nb_timestamps = len(times)
        str_repr += f"{nb_timestamps} timestamp{'' if nb_timestamps==1 else 's'}, "

        field_names = set()
        for time in times:
            field_names = field_names.union(self.get_field_names(time=time))
        nb_fields = len(field_names)
        str_repr += f"{nb_fields} field{'' if nb_fields==1 else 's'}, "

        # CGNS tree
        if self._meshes is None:
            str_repr += "no tree, "
        else:
            # TODO
            pass

        if str_repr[-2:] == ', ':
            str_repr = str_repr[:-2]
        str_repr = str_repr + ")"
        return str_repr

# %%
