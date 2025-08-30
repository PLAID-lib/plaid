"""Module for implementing collections of features within a Sample."""

import copy
import logging
from pathlib import Path
from typing import Optional

import CGNS.MAP as CGM
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import numpy as np
from CGNS.PAT.cgnsutils import __CHILDREN__, __NAME__

from plaid.constants import (
    CGNS_ELEMENT_NAMES,
)
from plaid.containers.utils import (
    _check_names,
    _read_index,
)
from plaid.types import CGNSLink, CGNSNode, CGNSTree, Field, Scalar
from plaid.utils import cgns_helper as CGH

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)


class SampleScalars:
    """A container for scalar features within a Sample.

    Provides dict-like operations for adding, retrieving, and removing scalars.

    Names must be unique and may not contain the character ``/``.

    Args:
        scalars (dict[str, Scalar], optional): a dict containing the pairs of (name, value) for each scalar in the `Sample`.
    """

    def __init__(self, scalars: Optional[dict[str, Scalar]]) -> None:
        self.data: dict[str, Scalar] = scalars if scalars is not None else {}

    def add(self, name: str, value: Scalar) -> None:
        """Add a scalar value to a dictionary.

        Args:
            name (str): The name of the scalar value.
            value (Scalar): The scalar value to add or update in the dictionary.
        """
        _check_names(name)
        self.data[name] = value

    def remove(self, name: str) -> Scalar:
        """Delete a scalar value from the dictionary.

        Args:
            name (str): The name of the scalar value to be deleted.

        Raises:
            KeyError: Raised when there is no scalar / there is no scalar with the provided name.

        Returns:
            Scalar: The value of the deleted scalar.
        """
        if name not in self.data:
            raise KeyError(f"There is no scalar value with name {name}.")

        return self.data.pop(name)

    def get(self, name: str) -> Optional[Scalar]:
        """Retrieve a scalar value associated with the given name.

        Args:
            name (str): The name of the scalar value to retrieve.

        Returns:
            Scalar or None: The scalar value associated with the given name, or None if the name is not found.
        """
        return self.data.get(name)

    def get_names(self) -> list[str]:
        """Get a set of scalar names available in the object.

        Returns:
            list[str]: A set containing the names of the available scalars.
        """
        return sorted(self.data.keys())


class SampleMeshes:
    """A container for meshes within a Sample."""

    def __init__(
        self,
        meshes: Optional[dict[float, CGNSTree]],
        mesh_base_name: str = "Base",
        mesh_zone_name: str = "Zone",
    ):
        self.data: dict[float, CGNSTree] = meshes if meshes is not None else {}

        self._default_active_base: Optional[str] = None
        self._default_active_zone: Optional[str] = None
        self._default_active_time: Optional[float] = None

        self._mesh_base_name: str = mesh_base_name
        self._mesh_zone_name: str = mesh_zone_name

    def get_all_mesh_times(self) -> list[float]:
        """Retrieve all time steps corresponding to the meshes, if available.

        Returns:
            list[float]: A list of all available time steps.
        """
        return list(self.data.keys())

    def get_time_assignment(self, time: Optional[float] = None) -> float:
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
        if self._default_active_time is None and time is None:
            timestamps = self.get_all_mesh_times()
            return sorted(timestamps)[0] if len(timestamps) > 0 else 0.0
        return self._default_active_time if time is None else time

    def get_base_assignment(
        self, base_name: Optional[str] = None, time: Optional[float] = None
    ) -> str:
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
        base_name = base_name or self._default_active_base

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
        self,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        zone_name = zone_name or self._default_active_zone

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

    def init_tree(self, time: Optional[float] = None) -> CGNSTree:
        """Initialize a CGNS tree structure at a specified time step or create a new one if it doesn't exist.

        Args:
            time (float, optional): The time step for which to initialize the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSTree (list): The initialized or existing CGNS tree structure for the specified time step.
        """
        time = self.get_time_assignment(time)

        if not self.data:
            self.data = {time: CGL.newCGNSTree()}
            self._links = {time: None}
            self._paths = {time: None}
        elif time not in self.data:
            self.data[time] = CGL.newCGNSTree()
            self._links[time] = None
            self._paths[time] = None

        return self.data[time]

    def get_links(self, time: Optional[float] = None) -> list[CGNSLink]:
        """Retrieve the CGNS links for a specified time step, if available.

        Args:
            time (float, optional): The time step for which to retrieve the CGNS links. If a specific time is not provided, the method will display the links for the default time step.

        Returns:
            list: The CGNS links for the specified time step if available; otherwise, returns None.
        """
        time = self.get_time_assignment(time)
        return self._links[time] if (self._links is not None) else None

    def get_mesh(
        self, time: Optional[float] = None, apply_links: bool = False, in_memory=False
    ) -> Optional[CGNSTree]:
        """Retrieve the CGNS tree structure for a specified time step, if available.

        Args:
            time (float, optional): The time step for which to retrieve the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.
            apply_links (bool, optional): Activates the following of the CGNS links to reconstruct the complete CGNS tree - in this case, a deepcopy of the tree is made to prevent from modifying the existing tree.
            in_memory (bool, optional): Active if apply_links == True, ONLY WORKING if linked mesh is in the current sample. This option follows the link in memory from current sample.

        Returns:
            CGNSTree: The CGNS tree structure for the specified time step if available; otherwise, returns None.
        """
        if self.data is None:
            return None

        time = self.get_time_assignment(time)
        tree = self.data[time]

        links = self.get_links(time)
        if not apply_links or links is None:
            return tree

        tree = copy.deepcopy(tree)
        for link in links:
            if not in_memory:
                subtree, _, _ = CGM.load(str(Path(link[0]) / link[1]), subtree=link[2])
            else:
                linked_timestep = int(link[1].split(".cgns")[0].split("_")[1])
                linked_timestamp = list(self.data.keys())[linked_timestep]
                subtree = self.get_mesh(linked_timestamp)
            node_path = "/".join(link[2].split("/")[:-1])
            node_to_append = CGU.getNodeByPath(tree, node_path)
            assert node_to_append is not None, (
                f"nodepath {node_path} not present in tree, cannot apply link"
            )
            node_to_append[2].append(CGU.getNodeByPath(subtree, link[2]))

        return tree

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

    def add_tree(self, tree: CGNSTree, time: Optional[float] = None) -> CGNSTree:
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

    # -------------------------------------------------------------------------#
    def get_topological_dim(
        self, base_name: Optional[str] = None, time: Optional[float] = None
    ) -> int:
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

    def get_physical_dim(
        self, base_name: Optional[str] = None, time: Optional[float] = None
    ) -> int:
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
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        self,
        full_path: bool = False,
        unique: bool = False,
        time: Optional[float] = None,
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

    def has_base(self, base_name: str, time: Optional[float] = None) -> bool:
        """Check if a CGNS tree contains a Base with a given name at a specified time.

        Args:
            base_name (str): The name of the Base to check for in the CGNS tree.
            time (float, optional): The time at which to check for the Base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            bool: `True` if the CGNS tree has a Base called `base_name`, else return `False`.
        """
        # get_base_names will look for the default time
        return base_name in self.get_base_names(time=time)

    def get_base(
        self, base_name: Optional[str] = None, time: Optional[float] = None
    ) -> CGNSNode:
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
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        base_name: Optional[str] = None,
        full_path: bool = False,
        unique: bool = False,
        time: Optional[float] = None,
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
        self,
        zone_name: str,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        self,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        self,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
    def get_nodal_tags(
        self,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
            indices = _read_index(BCNode, dim)
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
            indices = _read_index(ZSRNode, dim)
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
        self,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        self,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
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
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        location: str = "Vertex",
        time: Optional[float] = None,
    ) -> list[str]:
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
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        location: str = "Vertex",
        time: Optional[float] = None,
    ) -> Field:
        """Retrieve a field with a specified name from a given zone, base, location, and time.

        Args:
            name (str): The name of the field to retrieve.
            zone_name (str, optional): The name of the zone to search for. Defaults to None.
            base_name (str, optional): The name of the base to search for. Defaults to None.
            location (str, optional): The location at which to retrieve the field. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            time (float, optional): The time value to consider when searching for the field. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            Field: A set containing the names of the fields that match the specified criteria.
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
        field: Field,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        location: str = "Vertex",
        time: Optional[float] = None,
        warning_overwrite=True,
    ) -> None:
        """Add a field to a specified zone in the grid.

        Args:
            name (str): The name of the field to be added.
            field (Field): The field data to be added.
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
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        location: str = "Vertex",
        time: Optional[float] = None,
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

    def show_tree(self, time: Optional[float] = None) -> None:
        """Display the structure of the CGNS tree for a specified time.

        Args:
            time (float, optional): The time step for which you want to display the CGNS tree structure. Defaults to None. If a specific time is not provided, the method will display the tree structure for the default time step.
        """
        time = self.get_time_assignment(time)

        if self._meshes is not None:
            CGH.show_cgns_tree(self.data[time])
