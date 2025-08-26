"""Module for implementing collections."""

import logging
from typing import Optional, Union

import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import numpy as np

from plaid.types import CGNSNode, CGNSTree, Field, Scalar
from plaid.utils import cgns_helper as CGH

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)


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


class ScalarCollection:
    """Manager object for scalars."""

    def __init__(self):
        self.features: dict[str, Scalar] = {}

    def add(self, name: str, value: Scalar) -> None:
        """Add a scalar value to a dictionary.

        Args:
            name (str): The name of the scalar value.
            value (Scalar): The scalar value to add or update in the dictionary.
        """
        _check_names([name])
        if self._scalars is None:
            self._scalars = {name: value}
        else:
            self._scalars[name] = value

    def remove(self, name: str) -> Scalar:
        """Delete a scalar value from the dictionary.

        Args:
            name (str): The name of the scalar value to be deleted.

        Raises:
            KeyError: Raised when there is no scalar / there is no scalar with the provided name.

        Returns:
            Scalar: The value of the deleted scalar.
        """
        if self._scalars is None:
            raise KeyError("There is no scalar inside this sample.")

        if name not in self._scalars:
            raise KeyError(f"There is no scalar value with name {name}.")

        return self._scalars.pop(name)

    def get(self, name: str) -> Scalar:
        """Retrieve a scalar value associated with the given name.

        Args:
            name (str): The name of the scalar value to retrieve.

        Returns:
            Scalar or None: The scalar value associated with the given name, or None if the name is not found.
        """
        if (self._scalars is None) or (name not in self._scalars):
            return None
        else:
            return self._scalars[name]

    def get_names(self) -> list[str]:
        """Get a set of scalar names available in the object.

        Returns:
            set[str]: A set containing the names of the available scalars.
        """
        if self._scalars is None:
            return []
        else:
            res = sorted(self._scalars.keys())
            return res


class FieldCollection:
    """Manager object for fields."""

    def __init__(self):
        self.features: dict[str, Field] = {}
        self._defaults: dict = {
            "active_base": None,
            "active_zone": None,
            "active_time": None,
        }

    def add(
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
                CGL.newDataArray(flow_solution_node, name, np.asfortranarray(field))
            else:
                if warning_overwrite:
                    logger.warning(
                        f"field node with name {name} already exists -> data will be replaced"
                    )
                CGU.setValue(field_node, np.asfortranarray(field))

    def remove(
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

    def get(
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

    def get_names(
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
