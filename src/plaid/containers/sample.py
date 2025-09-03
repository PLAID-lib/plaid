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
from pydantic import BaseModel, model_serializer

from plaid.constants import (
    AUTHORIZED_FEATURE_INFOS,
    AUTHORIZED_FEATURE_TYPES,
    CGNS_FIELD_LOCATIONS,
)
from plaid.containers.features import SampleMeshes, SampleScalars, _check_names
from plaid.containers.utils import get_feature_type_and_details_from
from plaid.types import (
    CGNSLink,
    CGNSNode,
    CGNSPath,
    CGNSTree,
    Feature,
    FeatureIdentifier,
    Field,
    Scalar,
    TimeSequence,
    TimeSeries,
)
from plaid.utils import cgns_helper as CGH
from plaid.utils.base import safe_len

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)


class Sample(BaseModel):
    """Represents a single sample. It contains data and information related to a single observation or measurement within a dataset."""

    def __init__(
        self,
        directory_path: Optional[Union[str, Path]] = None,
        mesh_base_name: str = "Base",
        mesh_zone_name: str = "Zone",
        meshes: Optional[dict[float, CGNSTree]] = None,
        scalars: Optional[dict[str, Scalar]] = None,
        time_series: Optional[dict[str, TimeSeries]] = None,
        links: Optional[dict[float, list[CGNSLink]]] = None,
        paths: Optional[dict[float, list[CGNSPath]]] = None,
    ) -> None:
        """Initialize an empty :class:`Sample <plaid.containers.sample.Sample>`.

        Args:
            directory_path (Union[str, Path], optional): The path from which to load PLAID sample files.
            mesh_base_name (str, optional): The base name for the mesh. Defaults to 'Base'.
            mesh_zone_name (str, optional): The zone name for the mesh. Defaults to 'Zone'.
            meshes (dict[float, CGNSTree], optional): A dictionary mapping time steps to CGNSTrees. Defaults to None.
            scalars (dict[str, Scalar], optional): A dictionary mapping scalar names to their values. Defaults to None.
            time_series (dict[str, TimeSeries], optional): A dictionary mapping time series names to their values. Defaults to None.
            links (dict[float, list[CGNSLink]], optional): A dictionary mapping time steps to lists of links. Defaults to None.
            paths (dict[float, list[CGNSPath]], optional): A dictionary mapping time steps to lists of paths. Defaults to None.

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

        self._meshes = SampleMeshes(
            meshes, mesh_base_name, mesh_zone_name, links, paths
        )
        self._scalars = SampleScalars(scalars)
        self._time_series: Optional[dict[str, TimeSeries]] = time_series

        if directory_path is not None:
            directory_path = Path(directory_path)
            self.load(directory_path)

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

    def get_scalar(self, name: str) -> Optional[Scalar]:
        """Retrieve a scalar value associated with the given name.

        Args:
            name (str): The name of the scalar value to retrieve.

        Returns:
            Scalar or None: The scalar value associated with the given name, or None if the name is not found.
        """
        return self._scalars.get(name)

    def add_scalar(self, name: str, value: Scalar) -> None:
        """Add a scalar value to a dictionary.

        Args:
            name (str): The name of the scalar value.
            value (Scalar): The scalar value to add or update in the dictionary.
        """
        self._scalars.add(name, value)

    def del_scalar(self, name: str) -> Scalar:
        """Delete a scalar value from the dictionary.

        Args:
            name (str): The name of the scalar value to be deleted.

        Raises:
            KeyError: Raised when there is no scalar / there is no scalar with the provided name.

        Returns:
            Scalar: The value of the deleted scalar.
        """
        return self._scalars.remove(name)

    def get_scalar_names(self) -> list[str]:
        """Get a set of scalar names available in the object.

        Returns:
            list[str]: A set containing the names of the available scalars.
        """
        return self._scalars.get_names()

    # -------------------------------------------------------------------------#

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
        return self._meshes.get_mesh(time, apply_links, in_memory)

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
        if base_name in (self._meshes._default_active_base, None):
            return
        if not self._meshes.has_base(base_name, time):
            raise ValueError(f"base {base_name} does not exist at time {time}")

        self._meshes._default_active_base = base_name

    def set_default_zone_base(
        self, zone_name: str, base_name: str, time: Optional[float] = None
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
        if zone_name in (self._meshes._default_active_zone, None):
            return
        if not self._meshes.has_zone(zone_name, base_name, time):
            raise ValueError(
                f"zone {zone_name} does not exist for the base {base_name} at time {time}"
            )

        self._meshes._default_active_zone = zone_name

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
        return self._meshes.init_base(topological_dim, physical_dim, base_name, time)

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
        return self._meshes.init_zone(zone_shape, zone_type, zone_name, base_name, time)

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
        if time in (self._meshes._default_active_time, None):
            return
        if time not in self._meshes.get_all_mesh_times():
            raise ValueError(f"time {time} does not exist in mesh times")

        self._meshes._default_active_time = time

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
        return self._meshes.get_field_names(zone_name, base_name, location, time)

    # -------------------------------------------------------------------------#

    def link_tree(
        self,
        path_linked_sample: Union[str, Path],
        linked_sample: "Sample",
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

        if linked_time not in linked_sample._meshes.data:  # pragma: no cover
            raise KeyError(
                f"There is no CGNS tree for time {linked_time} in linked_sample."
            )
        if time in self._meshes.data:  # pragma: no cover
            raise KeyError(f"A CGNS tree is already linked in self for time {time}.")

        tree = CGL.newCGNSTree()

        base_names = linked_sample._meshes.get_base_names(time=linked_time)

        for bn in base_names:
            base_node = linked_sample._meshes.get_base(bn, time=linked_time)
            base = [bn, base_node[1], [], "CGNSBase_t"]
            tree[2].append(base)

            family = [
                "Bulk",
                np.array([b"B", b"u", b"l", b"k"], dtype="|S1"),
                [],
                "FamilyName_t",
            ]  # maybe get this from linked_sample as well ?
            base[2].append(family)

            zone_names = linked_sample._meshes.get_zone_names(bn, time=linked_time)
            for zn in zone_names:
                zone_node = linked_sample._meshes.get_zone(zn, bn, time=linked_time)
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
            Types_t = CGU.getAllNodesByTypeSet(sample._meshes.get_mesh(time), Type_t)
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

        self._meshes.add_tree(tree, time=time)

        dname = path_linked_sample.parent
        bname = path_linked_sample.name
        self._meshes._links[time] = [
            [str(dname), bname, fp, fp] for fp in feature_paths
        ]

        return tree

    def show_tree(self, time: Optional[float] = None) -> None:
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
        self._meshes.show_tree(time)

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
        self._meshes.add_field(
            name, field, zone_name, base_name, location, time, warning_overwrite
        )

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
        return self._meshes.get_field(name, zone_name, base_name, location, time)

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
        return self._meshes.del_field(name, zone_name, base_name, location, time)

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
        return self._meshes.get_nodes(zone_name, base_name, time)

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
        self._meshes.set_nodes(nodes, zone_name, base_name, time)

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

    def get_time_series(self, name: str) -> Optional[TimeSeries]:
        """Retrieve a time series by name.

        Args:
            name (str): The name of the time series to retrieve.

        Returns:
            TimeSeries or None: If a time series with the given name exists, it returns the corresponding time series, or None otherwise.

        """
        if (self._time_series is None) or (name not in self._time_series):
            return None
        else:
            return self._time_series[name]

    def add_time_series(
        self, name: str, time_sequence: TimeSequence, values: Field
    ) -> None:
        """Add a time series to the sample.

        Args:
            name (str): A descriptive name for the time series.
            time_sequence (TimeSequence): The time sequence, array of time points.
            values (Field): The values corresponding to the time sequence.

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

    def del_time_series(self, name: str) -> tuple[TimeSequence, Field]:
        """Delete a time series from the sample.

        Args:
            name (str): The name of the time series to be deleted.

        Raises:
            KeyError: Raised when there is no time series / there is no time series with the provided name.

        Returns:
            tuple[TimeSequence, Field]: A tuple containing the time sequence and values of the deleted time series.
        """
        if self._time_series is None:
            raise KeyError("There is no time series inside this sample.")

        if name not in self._time_series:
            raise KeyError(f"There is no time series with name {name}.")

        return self._time_series.pop(name)

    # -------------------------------------------------------------------------#

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
        for t in self._meshes.get_all_mesh_times():
            for bn in self._meshes.get_base_names(time=t):
                for zn in self._meshes.get_zone_names(base_name=bn, time=t):
                    if (
                        self._meshes.get_nodes(base_name=bn, zone_name=zn, time=t)
                        is not None
                    ):
                        all_features_identifiers.append(
                            {
                                "type": "nodes",
                                "base_name": bn,
                                "zone_name": zn,
                                "time": t,
                            }
                        )
                    for loc in CGNS_FIELD_LOCATIONS:
                        for fn in self._meshes.get_field_names(
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
    ) -> Feature:
        """Retrieve a specific feature from its encoded string identifier.

        The `feature_string_identifier` must follow the format:
            "<feature_type>::<detail1>/<detail2>/.../<detailN>"

        Supported feature types:
            - "scalar": expects 1 detail → `scalars.get(name)`
            - "time_series": expects 1 detail → `get_time_series(name)`
            - "field": up to 5 details → `get_field(name, base_name, zone_name, location, time)`
            - "nodes": up to 3 details → `get_nodes(base_name, zone_name, time)`

        Args:
            feature_string_identifier (str): Structured identifier of the feature.

        Returns:
            Feature: The retrieved feature object.

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
            val = self.get_scalar(feature_details[0])
            if val is None:
                raise KeyError(
                    f"Unknown scalar {feature_details[0]}"
                )  # pragma: no cover
            return val
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
    ) -> Feature:
        """Retrieve a feature object based on a structured identifier dictionary.

        The `feature_identifier` must include a `"type"` key specifying the feature kind:
            - `"scalar"`       → calls `scalars.get(name)`
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
            Feature: The corresponding feature instance retrieved via the appropriate accessor.
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
    ) -> list[Feature]:
        """Retrieve features based on a list of structured identifier dictionaries.

        Elements of `feature_identifiers` must include a `"type"` key specifying the feature kind:
            - `"scalar"`       → calls `scalars.get(name)`
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
            list[Feature]: List of corresponding feature instance retrieved via the appropriate accessor.
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
        feature: Feature,
    ) -> Self:
        """Add a feature to current sample.

        This method applies updates to scalars, time series, fields, or nodes
        using feature identifiers, and corresponding feature data.

        Args:
            feature_identifier (dict): A feature identifier.
            feature (Feature): A feature corresponding to the identifiers.

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
            phys_dim = self._meshes.get_physical_dim(**physical_dim_arg)
            self.set_nodes(**feature_details, nodes=feature.reshape((-1, phys_dim)))

        return self

    def update_features_from_identifier(
        self,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
        features: Union[Feature, list[Feature]],
        in_place: bool = False,
    ) -> Self:
        """Update one or several features of the sample by their identifier(s).

        This method applies updates to scalars, time series, fields, or nodes
        using feature identifiers, and corresponding feature data. When `in_place=False`, a deep copy of the sample is created
        before applying updates, ensuring full isolation from the original.

        Args:
            feature_identifiers (dict or list of dict): One or more feature identifiers.
            features (Feature or list of Feature): One or more features corresponding
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

            if feature is not None:
                # if trying to add a field or nodes, must check if the corresponding tree exists, and add it if not
                if feat_id["type"] in ["field", "nodes"]:
                    # get time of current feature
                    time = self._meshes.get_time_assignment(time=feat_id.get("time"))

                    # if the constructed sample does not have a tree, add the one from the source sample, with no field
                    if not sample._meshes.get_mesh(time):
                        sample._meshes.add_tree(source_sample._meshes.get_mesh(time))

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
                time = sample._meshes.get_time_assignment(time=feat_id.get("time"))

                # if the constructed sample does not have a tree, add the one from the source sample, with no field
                if not merged_dataset._meshes.get_mesh(time):
                    merged_dataset._meshes.add_tree(source_sample.get_mesh(time))

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

        if self._meshes.data:
            mesh_dir.mkdir()
            for i, time in enumerate(self._meshes.data.keys()):
                outfname = mesh_dir / f"mesh_{i:09d}.cgns"
                status = CGM.save(
                    str(outfname),
                    self._meshes.data[time],
                    links=self._meshes._links.get(time),
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
            # self._meshes = {}
            self._meshes._links = {}
            self._meshes._paths = {}
            for i in range(nb_meshes):
                tree, links, paths = CGM.load(str(meshes_dir / f"mesh_{i:09d}.cgns"))
                time = CGH.get_time_values(tree)

                (
                    self._meshes.data[time],
                    self._meshes._links[time],
                    self._meshes._paths[time],
                ) = (
                    tree,
                    links,
                    paths,
                )
                for i in range(len(self._meshes._links[time])):  # pragma: no cover
                    self._meshes._links[time][i][0] = str(
                        meshes_dir / self._meshes._links[time][i][0]
                    )

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
        times = self._meshes.get_all_mesh_times()
        nb_timestamps = len(times)
        str_repr += f"{nb_timestamps} timestamp{'' if nb_timestamps == 1 else 's'}, "

        field_names = set()
        for time in times:
            ## Need to include all possible location within the count
            base_names = self._meshes.get_base_names(time=time)
            for bn in base_names:
                zone_names = self._meshes.get_zone_names(base_name=bn)
                for zn in zone_names:
                    field_names = field_names.union(
                        self._meshes.get_field_names(
                            zone_name=zn, time=time, location="Vertex"
                        )
                        + self._meshes.get_field_names(
                            zone_name=zn, time=time, location="EdgeCenter"
                        )
                        + self._meshes.get_field_names(
                            zone_name=zn, time=time, location="FaceCenter"
                        )
                        + self._meshes.get_field_names(
                            zone_name=zn, time=time, location="CellCenter"
                        )
                    )
        nb_fields = len(field_names)
        str_repr += f"{nb_fields} field{'' if nb_fields == 1 else 's'}, "

        # CGNS tree
        if not self._meshes.data:
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
            "mesh_base_name": self._meshes._mesh_base_name,
            "mesh_zone_name": self._meshes._mesh_zone_name,
            "meshes": self._meshes.data,
            "scalars": self._scalars.data,
            "time_series": self._time_series,
            "links": self._meshes._links,
            "paths": self._meshes._paths,
        }
