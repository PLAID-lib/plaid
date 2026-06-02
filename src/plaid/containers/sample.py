"""Implementation of the `Sample` container."""

# %% Imports
import sys
from typing import Sequence

from ..types.cgns_types import CGNSTree
from ..types.common import ScalarDType, ScalarOrArrayOrStr

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import logging
import pickle
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import CGNS.MAP as CGM
import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import numpy as np
from CGNS.PAT.cgnsutils import __CHILDREN__, __NAME__
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as PydanticField

from ..constants import (
    AUTHORIZED_FEATURE_TYPES,
    AUTHORIZED_FEATURE_TYPES_T,
    CGNS_ELEMENT_NAMES,
    CGNS_FIELD_LOCATIONS,
)
from ..types import Array, CGNSNode
from ..utils import cgns_helper as CGH
from ..utils.base import safe_len
from .managers.default_manager import DefaultManager
from .utils import _check_names, _read_index, get_feature_details_from_path

logger = logging.getLogger(__name__)

CGNS_WORKER = Path(__file__).parent.parent / "utils" / "cgns_worker.py"


class Sample(BaseModel):
    """Represents a single sample. It contains data and information related to a single observation or measurement within a dataset.

    By default, the sample is empty but:
        - You can provide a path to a folder containing the sample data, and it will be loaded during initialization.

    Note:
        Mesh/field/global operations are directly implemented on ``Sample`` via
        inheritance from internal feature operations.
    """

    # Pydantic configuration
    # TODO(FB) check why arbitrary_types_allowed is needed, and if it can be removed
    model_config = ConfigDict(
        arbitrary_types_allowed=True, revalidate_instances="always", extra="forbid"
    )

    # Attributes
    path: Optional[Union[str, Path]] = PydanticField(
        None,
        description="Path to the folder containing the sample data. If provided, the sample will be loaded from this path during initialization. Defaults to None.",
    )

    data: dict[float, CGNSTree] = PydanticField(
        default_factory=dict,
        description="A dictionary mapping time steps to CGNS trees.",
    )

    defaults: DefaultManager = PydanticField(
        default=None,
        exclude=True,
        repr=False,
        description="Default selector manager (time/base/zone).",
    )

    @model_validator(mode="after")
    def initialize_defaults(self) -> "Sample":
        """Initialize the default manager if not already set."""
        if self.defaults is None:
            self.defaults = DefaultManager(self)
        return self

    def model_post_init(self, _context: Any) -> None:
        """Run post-initialization hooks (e.g. load sample from path)."""
        # Load if path is provided
        if self.path is not None:
            path = Path(self.path)
            self.load(path)

    def copy(self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create a deep copy of the current `Sample` instance.

        Usage of `model_copy(deep=True)` from Pydantic to ensure all internal data is deeply copied.

        Returns:
            A new `Sample` instance with all internal data (scalars, fields, meshes, etc.)
            deeply copied to ensure full isolation from the original.

        Note:
            This operation may be memory-intensive for large samples.
        """
        return self.model_copy(deep=True)

    def get_all_features_identifiers_by_type(
        self, feature_type: AUTHORIZED_FEATURE_TYPES_T
    ) -> list[str]:
        """Get all features identifiers of a given type from the sample.

        Args:
            feature_type (str): Type of features to return

        Returns:
            list[FeatureIdentifier]: A list of dictionaries containing the identifiers of a given type of all features in the sample.
        """
        assert feature_type in AUTHORIZED_FEATURE_TYPES, "feature_type not known"
        if feature_type == "scalar":
            return self.get_global_names()
        elif feature_type == "field":
            return self.get_field_names()
        elif feature_type == "nodes":
            return [
                "Coordinate" + n
                for _, n in zip(range(self.get_physical_dim()), ["X", "Y", "Z"])
            ]

    def get_all_features_by_type(self, type: str) -> list[str]:
        """Get the list of all CGNS paths of features of a given type (eg 'field', 'global', 'coordinate', etc...)."""
        flat_tree, _ = CGH.flatten_cgns_tree(self.get_tree())
        out = []
        for path in flat_tree:
            feature_details = get_feature_details_from_path(path)
            if feature_details["type"] == type:
                if type == "global":
                    if feature_details["sub_type"] == "scalar":
                        out.append(path)
                else:
                    out.append(path)
        return out

    def get_feature_by_path(
        self, path: str, time: Optional[float | np.floating] = None
    ) -> np.number | np.ndarray | None:
        """Retrieve a feature value from the sample's CGNS mesh using a CGNS-style url.

        Args:
            path (str): CGNS node path relative to the mesh root (for example
                "BaseName/ZoneName/GridCoordinates/CoordinateX" or
                "BaseName/ZoneName/Solution/FieldName").
            time (Optional[float | np.floating], optional): Time selection for the mesh. If an integer,
                it is interpreted via the sample time-assignment logic
                (see ``resolve_time``). If None, the default time
                assignment is used. Defaults to None.

        Returns:
            Feature: The value stored at the given CGNS path. This may be a numpy array, a scalar, or None if the node has no value.

        Note:
            - This is a thin wrapper around CGNS.PAT.cgnsutils.getValueByPath and Sample.get_tree(time). Callers should handle a returned None when the path or value does not exist.
            - For field-like features, prefer using Sample.get_field which applies additional validation and selection logic.
        """
        time = self.resolve_time(time)
        return CGU.getValueByPath(self.get_tree(time), path)

    def add_feature(
        self,
        feature_path: str,
        feature: ScalarOrArrayOrStr,
    ) -> Self:
        """Add a feature to current sample.

        This method applies updates to scalars, fields, or nodes
        using feature identifiers, and corresponding feature data.

        Args:
            feature_path (str): A feature path string.
            feature (Feature): Feature value corresponding to ``feature_path``.

        Returns:
            Self: The updated sample

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """
        # feature_type, feature_details = get_feature_type_and_details_from(
        #    feature_identifier
        # )

        from .utils import get_feature_details_from_path

        feature_details = get_feature_details_from_path(feature_path)

        feature_type = feature_details.pop("type")
        _ = feature_details.pop("sub_type", None)

        if feature_type == "global":
            if safe_len(feature) == 1:
                feature = feature[0]
            self.add_global(**feature_details, global_array=feature)
        elif feature_type == "field":
            self.add_field(**feature_details, field=feature, warning_overwrite=False)
        elif feature_type == "coordinate":
            if feature_details.get("name", None) is not None:  # pragma: no cover
                raise ValueError("Must set the 3 coordinate at the same time")
            physical_dim_arg = {
                k: v for k, v in feature_details.items() if k in ["base", "time"]
            }
            phys_dim = self.get_physical_dim(**physical_dim_arg)
            self.set_nodes(**feature_details, nodes=feature.reshape((-1, phys_dim)))
        else:  # pragma: no cover
            print(feature_details)
            raise RuntimeError(f"feature_type not allowed : {feature_type}")

        return self

    def del_feature_by_path(self, path: str, time: Optional[float] = None) -> CGNSTree:
        """Delete a feature/node by CGNS-style path from the sample mesh tree.

        Args:
            path (str): CGNS node path relative to the mesh root (for example
                "BaseName/ZoneName/GridCoordinates/CoordinateX" or
                "BaseName/ZoneName/Solution/FieldName").
            time (Optional[int], optional): Time selection for the mesh. If an integer,
                it is interpreted via the sample time-assignment logic
                (see ``resolve_time``). If None, the default time
                assignment is used. Defaults to None.

        Returns:
            Feature: Updated tree after node deletion.

        Note:
            - This method resolves the requested time and deletes the node at
              ``path`` when present.
        """
        time = self.resolve_time(time)
        #        return CGU.getValueByPath(self.get_tree(time), path)

        updated_tree = None
        node = CGU.getNodeByPath(self.get_tree(time), path)
        if node is not None:
            updated_tree = CGU.nodeDelete(self.get_tree(time), node)

        # If the function reaches here, the field was not found
        if updated_tree is None:  # pragma: no cover
            raise KeyError(f"There is no field with name {path} in the specified zone.")

        return updated_tree

    def update_features_by_path(
        self,
        feature_identifiers: str | Sequence[str],
        features: Union[ScalarDType, list[ScalarDType]],
        in_place: bool = False,
    ) -> Self:
        """Update one or several features of the sample by their identifier(s).

        This method applies updates to scalars, fields, or nodes
        using feature identifiers, and corresponding feature data. When `in_place=False`, a deep copy of the sample is created
        before applying updates, ensuring full isolation from the original.

        Args:
            feature_identifiers (FeatureIdentifier or list of FeatureIdentifier): One or more feature identifiers.
            features (dict of Feature or list of Feature): One or more features corresponding
                to the identifiers.
            in_place (bool, optional): If True, modifies the current sample in place.
                If False, returns a deep copy with updated features.

        Returns:
            Self: The updated sample (either the current instance or a new copy).

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """
        if not isinstance(feature_identifiers, list):
            feature_identifiers = [feature_identifiers]
            features = [features]
        assert len(feature_identifiers) == len(features)
        for i_id, feat_id in enumerate(feature_identifiers):
            feature_identifiers[i_id] = str(feat_id)

        sample = self if in_place else self.copy()

        for feat_id, feat in zip(feature_identifiers, features):
            sample.add_feature(feat_id, feat)

        return sample

    # -------------------------------------------------------------------------#
    def save_to_dir(
        self, path: Union[str, Path], overwrite: bool = False, memory_safe: bool = False
    ) -> None:
        """Save the Sample in directory `path`.

        Args:
            path (Union[str,Path]): relative or absolute directory path.
            overwrite (bool): target directory overwritten if True.
            memory_safe (bool): use pyCGNS save in a subprocess (requires an additional pickle of the sample) if True.
        """
        path = Path(path)

        if path.is_dir():
            if overwrite:
                shutil.rmtree(path)
                logger.warning(f"Existing {path} directory has been reset.")
            elif any(path.iterdir()):
                raise ValueError(
                    f"directory {path} already exists and is not empty. Set `overwrite` to True if needed."
                )

        path.mkdir(exist_ok=True)

        mesh_dir = path / "meshes"

        if self.data:
            mesh_dir.mkdir()
            for i, time in enumerate(self.data.keys()):
                outfname = mesh_dir / f"mesh_{i:09d}.cgns"
                if memory_safe:
                    tmpfile = mesh_dir / f"mesh_{i:09d}.pkl"
                    with open(tmpfile, "wb") as f:
                        pickle.dump(self.data[time], f)

                    cmd = [sys.executable, str(CGNS_WORKER), tmpfile, str(outfname)]
                    subprocess.run(cmd)
                    logger.debug(f"save -> {outfname}")

                else:
                    status = CGM.save(str(outfname), self.data[time])
                    logger.debug(f"save -> {status=}")

    @classmethod
    def load_from_dir(cls, path: Union[str, Path]) -> Self:
        """Load the Sample from directory `path`.

        This is a class method, you don't need to instantiate a `Sample` first.

        Args:
            path (Union[str,Path]): Relative or absolute directory path.

        Returns:
            Sample

        Example:
            .. code-block:: python

                from plaid import Sample
                sample = Sample.load_from_dir(dir_path)
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)

        Note:
            It calls :meth:`Sample.load` method during execution.
        """
        path = Path(path)
        instance = cls()
        instance.load(path)
        return instance

    def load(self, path: Union[str, Path]) -> None:
        """Load the Sample from directory `path`.

        Args:
            path (Union[str,Path]): Relative or absolute directory path.

        Raises:
            FileNotFoundError: Triggered if the provided directory does not exist.
            FileExistsError: Triggered if the provided path is a file instead of a directory.

        Example:
            .. code-block:: python

                from plaid import Sample
                sample = Sample()
                sample.load(path)
                print(sample)
                >>> Sample(3 scalars, 1 timestamp, 3 fields)

        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f'Directory "{path}" does not exist. Abort')

        if not path.is_dir():
            raise FileExistsError(f'"{path}" is not a directory. Abort')

        meshes_dir = path / "meshes"
        if meshes_dir.is_dir():
            meshes_names = list(meshes_dir.glob("*"))
            nb_meshes = len(meshes_names)
            for i in range(nb_meshes):
                tree, _, _ = CGM.load(str(meshes_dir / f"mesh_{i:09d}.cgns"))
                time = CGH.get_time_values(tree)

                (self.data[time],) = (tree,)

    # # -------------------------------------------------------------------------#
    def __str__(self) -> str:
        """Return a string representation of the sample.

        Returns:
            str: A string representation of the overview of sample content.
        """
        # TODO rewrite using self.get_all_features_identifiers()
        str_repr = "Sample("

        # globals
        nb_globals = len(self.get_global_names())
        str_repr += f"{nb_globals} global{'' if nb_globals == 1 else 's'}, "

        # fields
        times = self.get_all_time_values()
        nb_timestamps = len(times)
        str_repr += f"{nb_timestamps} timestamp{'' if nb_timestamps == 1 else 's'}, "

        field_names = set()
        for time in times:
            ## Need to include all possible location within the count
            base_names = self.get_base_names(time=time)
            for bn in base_names:
                zone_names = self.get_zone_names(base=bn, time=time)
                for zn in zone_names:
                    for location in CGNS_FIELD_LOCATIONS:
                        field_names = field_names.union(
                            self.get_field_names(
                                location=location, zone=zn, base=bn, time=time
                            )
                        )
        nb_fields = len(field_names)
        str_repr += f"{nb_fields} field{'' if nb_fields == 1 else 's'}, "

        if str_repr[-2:] == ", ":
            str_repr = str_repr[:-2]
        str_repr = str_repr + ")"

        return str_repr

    __repr__ = __str__

    def summarize(self) -> str:
        """Provide detailed summary of the Sample content, showing feature names and mesh information.

        This provides more detailed information than the __repr__ method,
        including the name of each feature.

        Returns:
            str: A detailed string representation of the sample content.

        Example:
            .. code-block:: bash

                Sample Summary:
                ==================================================
                Scalars (8):
                - Pr: 0.9729006564945664
                - Q: 0.2671142611487964
                - Tr: 0.9983394202616822
                - angle_in: 45.5066666666667
                - angle_out: 61.89519547386746
                - eth_is: 0.21238326882538008
                - mach_out: 0.81003
                - power: 0.0019118127462776008

                Meshes (1 timestamps):
                Time: 0.0
                    Base: Base_2_2
                        Nodes (36421)
                        Tags (6): Intrado (122), Extrado (122), Inflow (121), Outflow (121), Periodic_1 (120), Periodic_2 (238)
                        Fields (7): ro, sdf, rou, nut, mach, roe, rov
                        Elements (36000)
                        QUAD_4 (36000)
                    Base: Base_1_2
                        Nodes (244)
                        Fields (1): M_iso
                        Elements (242)
                        BAR_2 (242)
        """
        summary = "Sample Summary:\n"
        summary += "=" * 50 + "\n"

        # Scalars with names
        scalar_names = self.get_global_names()
        if scalar_names:
            summary += f"Scalars ({len(scalar_names)}):\n"
            for name in scalar_names:
                value = self.get_global(name)
                summary += f"  - {name}: {value}\n"
            summary += "\n"

        # Mesh information
        times = self.get_all_time_values()
        summary += f"Meshes ({len(times)} timestamps):\n"
        if times:
            for time in times:
                summary += f"    Time: {time}\n"
                base_names = self.get_base_names(time=time)
                for base_name in base_names:
                    summary += f"        Base: {base_name}\n"
                    zone_names = self.get_zone_names(base=base_name, time=time)
                    for zone_name in zone_names:
                        summary += f"            Zone: {zone_name}\n"
                        # Nodes, nodal tags and fields at verticies
                        nodes = self.get_nodes(
                            zone=zone_name, base=base_name, time=time
                        )
                        if nodes is not None:
                            nb_nodes = nodes.shape[0]
                            nodal_tags = self.get_nodal_tags(
                                zone=zone_name, base=base_name, time=time
                            )
                            summary += f"                Nodes ({nb_nodes})\n"
                            if len(nodal_tags) > 0:
                                summary += f"                Tags ({len(nodal_tags)}): {', '.join([f'{k} ({len(v)})' for k, v in nodal_tags.items()])}\n"

                        for location in CGNS_FIELD_LOCATIONS:
                            field_names = self.get_field_names(
                                location=location,
                                zone=zone_name,
                                base=base_name,
                                time=time,
                            )
                            if field_names:
                                summary += f"                Location: {location}\n                    Fields ({len(field_names)}): {', '.join(field_names)}\n"

                        # Elements and fields at elements
                        elements = self.get_elements(
                            zone=zone_name, base=base_name, time=time
                        )
                        summary += f"                Elements ({sum([v.shape[0] for v in elements.values()])})\n"
                        if len(elements) > 0:
                            summary += f"                    {', '.join([f'{k} ({v.shape[0]})' for k, v in elements.items()])}\n"

        return summary

    def check_completeness(self) -> str:
        """Check the completeness of features in this sample.

        Returns:
            str: A report on feature completeness.

        Example:
            .. code-block:: bash

                Sample Completeness Check:
                ==============================
                Has scalars: True
                Has meshes: True
                Total unique fields: 8
                Field names: M_iso, mach, nut, ro, roe, rou, rov, sdf
        """
        report = "Sample Completeness Check:\n"
        report += "=" * 30 + "\n"

        # Check if sample has basic features
        has_scalars = len(self.get_global_names()) > 0
        has_meshes = len(self.get_all_time_values()) > 0

        report += f"Has scalars: {has_scalars}\n"
        report += f"Has meshes: {has_meshes}\n"

        if has_meshes:
            times = self.get_all_time_values()
            total_fields = set()
            for time in times:
                base_names = self.get_base_names(time=time)
                for base_name in base_names:
                    zone_names = self.get_zone_names(base=base_name, time=time)
                    for zone_name in zone_names:
                        for location in CGNS_FIELD_LOCATIONS:
                            field_names = self.get_field_names(
                                location=location,
                                zone=zone_name,
                                base=base_name,
                                time=time,
                            )
                            total_fields.update(field_names)

            report += f"Total unique fields: {len(total_fields)}\n"
            if total_fields:
                report += f"Field names: {', '.join(sorted(total_fields))}\n"

        return report

    # -------------------------------------------------------------------------#
    # Default time/base/zone management interface
    # -------------------------------------------------------------------------#

    def set_default_time(self, time: float) -> None:
        """Set the default active time. Calls the DefaultManager to set the default time.

        Args:
            time (float): The time to set as the default active time.
        """
        self.defaults.set_default_time(time)

    def set_default_base(self, base: str, time: Optional[float] = None) -> None:
        """Set the default active base. Calls the DefaultManager to set the default base.

        Args:
            base (str): The base name to set as the default active base.
            time (float, optional): The time at which to set the default base. Defaults to None.
        """
        self.defaults.set_default_base(base, time=time)

    def set_default_zone_base(
        self, zone: str, base: str, time: Optional[float] = None
    ) -> None:
        """Set the default active zone within a base. Calls the DefaultManager to set the default zone and base.

        Args:
            zone (str): The zone name to set as the default active zone.
            base (str): The base name in which the zone is located.
            time (float, optional): The time at which to set the default zone and base. Defaults to None.
        """
        self.defaults.set_default_zone_base(zone, base, time=time)

    def resolve_time(self, time: Optional[float] = None) -> float:
        """Get the resolved time assignment. Calls the DefaultManager to resolve the time.

        Args:
            time (float, optional): The time to resolve. Defaults to None.

        Returns:
            float: The resolved time.
        """
        return self.defaults.resolve_time(time)

    def resolve_base(
        self, base: Optional[str] = None, time: Optional[float] = None
    ) -> Optional[str]:
        """Get the resolved base assignment. Calls the DefaultManager to resolve the base.

        Args:
            base (str, optional): The base name to resolve. Defaults to None.
            time (float, optional): The time at which to resolve the base. Defaults to None.

        Returns:
            Optional[str]: The resolved base name.
        """
        return self.defaults.resolve_base(base=base, time=time)

    def resolve_zone(
        self,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> Optional[str]:
        """Get the resolved zone assignment. Calls the DefaultManager to resolve the zone.

        Args:
            zone (str, optional): The zone name to resolve. Defaults to None.
            base (str, optional): The base name in which the zone is located. Defaults to None.
            time (float, optional): The time at which to resolve the zone. Defaults to None.

        Returns:
            Optional[str]: The resolved zone name.
        """
        return self.defaults.resolve_zone(zone=zone, base=base, time=time)

    # -------------------------------------------------------------------------#

    def get_all_time_values(self) -> list[float]:
        """Retrieve all time steps corresponding to the meshes, if available.

        Returns:
            list[float]: A list of all available time steps.
        """
        return list(self.data.keys())

    def init_tree(self, time: Optional[float] = None) -> CGNSTree:
        """Initialize a CGNS tree structure at a specified time step or create a new one if it doesn't exist.

        Args:
            time (float, optional): The time step for which to initialize the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSTree (list): The initialized or existing CGNS tree structure for the specified time step.
        """
        time = self.resolve_time(time)

        if not self.data:
            self.data = {time: CGL.newCGNSTree()}
        elif time not in self.data:
            self.data[time] = CGL.newCGNSTree()

        return self.data[time]

    def get_tree(
        self, time: Optional[float] = None, only_mesh: bool = False
    ) -> Optional[CGNSTree]:
        """Retrieve the CGNS tree structure for a specified time step, if available.

        Args:
            time (float, optional): The time step for which to retrieve the CGNS tree structure. If a specific time is not provided, the method will display the tree structure for the default time step.
            only_mesh (bool): If True, features of type global and fields are removed from the sample

        Returns:
            CGNSTree: The CGNS tree structure for the specified time step if available; otherwise, returns None.
        """
        if not self.data:
            return None

        time = self.resolve_time(time)
        tree = self.data[time]
        if only_mesh:
            flat_tree, cgns_types = CGH.flatten_cgns_tree(tree)
            updated_flat_tree = {
                path: value
                for path, value in flat_tree.items()
                if get_feature_details_from_path(path)["type"]
                not in ["global", "field"]
            }
            tree = CGH.unflatten_cgns_tree(updated_flat_tree, cgns_types)
        return tree

    def set_trees(self, meshes: dict[float, CGNSTree]) -> None:
        """Set all meshes with their corresponding time step.

        Args:
            meshes (dict[float,CGNSTree]): Collection of time step with its corresponding CGNSTree.

        Raises:
            KeyError: If there is already a CGNS tree set.
        """
        if not self.data:
            self.data = meshes
        else:
            raise KeyError(
                "meshes is already set, you cannot overwrite it, delete it first or extend it with `Sample.add_tree`"
            )

    def add_tree(
        self, tree: CGNSTree, time: Optional[float] = None, in_place: bool = True
    ) -> CGNSTree:
        """Merge a CGNS tree into the tree stored at a given time.

        If there is no tree at ``time``, ``tree`` is stored directly. Otherwise, Base
        nodes from ``tree`` are appended only when their name does not already exist in
        the destination tree. Bases with duplicate names are ignored and a warning is
        emitted.

        Args:
            tree (CGNSTree): CGNS tree to add.
            time (float, optional): Time step at which the tree is added. If omitted,
                the default time resolution is used.
            in_place (bool, optional): Controls ownership of the input tree. When
                ``True`` (default), the provided object may be stored/used directly.
                When ``False``, the input tree is deep-copied before insertion.

        Raises:
            ValueError: If ``tree`` is an empty list.

        Returns:
            CGNSTree: The resulting tree for the resolved ``time``.
        """
        if tree == []:
            raise ValueError("CGNS Tree should not be an empty list")

        if not in_place:
            tree = deepcopy(tree)

        def _iter_node_names(node: CGNSNode) -> list[str]:
            names = []
            if isinstance(node, list) and len(node) > 0:
                if isinstance(node[0], str):
                    names.append(node[0])
                if len(node) > 2 and isinstance(node[2], list):
                    for child in node[2]:
                        names.extend(_iter_node_names(child))
            return names

        all_names = _iter_node_names(tree)
        if all_names and all_names[0] == "CGNSTree":
            all_names = all_names[1:]
        _check_names(all_names)

        time = self.resolve_time(time)

        if not self.data:
            self.data = {time: tree}
        elif time not in self.data:
            self.data[time] = tree
        else:
            # TODO: gérer le cas où il y a des bases de mêmes noms... + merge
            # récursif des nœuds
            local_bases = self.get_base_names(time=time)
            base_nodes = CGU.getNodesFromTypeSet(tree, "CGNSBase_t")
            for _, node in base_nodes:
                if node[__NAME__] not in local_bases:  # pragma: no cover
                    self.data[time][__CHILDREN__].append(node)
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

        return self.data[time]

    def del_tree(self, time: float) -> CGNSTree:
        """Delete the CGNS tree for a specific time.

        Args:
            time (float): The time step for which to delete the CGNS tree structure.

        Raises:
            KeyError: There is no CGNS tree in this Sample / There is no CGNS tree for the provided time.

        Returns:
            CGNSTree: The deleted CGNS tree.
        """
        if not self.data:
            raise KeyError("There is no CGNS tree in this sample.")

        if time not in self.data:
            raise KeyError(f"There is no CGNS tree for time {time}.")

        return self.data.pop(time)

    # -------------------------------------------------------------------------#
    def get_topological_dim(
        self, base: Optional[str] = None, time: Optional[float] = None
    ) -> int:
        """Get the topological dimension of a base node at a specific time.

        Args:
            base (str, optional): The name of the base node for which to retrieve the topological dimension. Defaults to None.
            time (float, optional): The time at which to retrieve the topological dimension. Defaults to None.

        Raises:
            ValueError: If there is no base node with the specified `base` at the given `time` in this sample.

        Returns:
            int: The topological dimension of the specified base node at the given time.
        """
        # get_base will look for default time and base
        base_node = self.get_base(base, time)

        if base_node is None:  # pragma: no cover
            raise ValueError(
                f"There is no base called {base} at the time {time} in this sample."
            )

        return int(base_node[1][0])

    def get_physical_dim(
        self, base: Optional[str] = None, time: Optional[float] = None
    ) -> int:
        """Get the physical dimension of a base node at a specific time.

        Args:
            base (str, optional): The name of the base node for which to retrieve the topological dimension. Defaults to None.
            time (float, optional): The time at which to retrieve the topological dimension. Defaults to None.

        Raises:
            ValueError: If there is no base node with the specified `base` at the given `time` in this sample.

        Returns:
            int: The topological dimension of the specified base node at the given time.
        """
        base_node = self.get_base(base, time)
        if base_node is None:  # pragma: no cover
            raise ValueError(
                f"There is no base called {base} at the time {time} in this sample."
            )

        return int(base_node[1][1])

    def init_base(
        self,
        topological_dim: int,
        physical_dim: int,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> CGNSNode:
        """Create a Base node named `base` if it doesn't already exists.

        Args:
            topological_dim (int): Cell dimension, see [CGNS standard](https://pycgns.github.io/PAT/lib.html#CGNS.PAT.cgnslib.newCGNSBase).
            physical_dim (int): Ambient space dimension, see [CGNS standard](https://pycgns.github.io/PAT/lib.html#CGNS.PAT.cgnslib.newCGNSBase).
            base (str): If not specified, uses `mesh_base_name` specified in Sample initialization. Defaults to None.
            time (float, optional): The time at which to initialize the base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSNode: The created Base node.
        """
        _check_names([base])

        time = self.resolve_time(time)

        if base is None:
            base = "Base_" + str(topological_dim) + "_" + str(physical_dim)

        self.init_tree(time)
        if not (self.has_base(base, time)):
            base_node = CGL.newCGNSBase(
                self.data[time], base, topological_dim, physical_dim
            )

        base_names = self.get_base_names(time=time)
        for base in base_names:
            base_node = self.get_base(base, time=time)
            if CGU.getValueByPath(base_node, "Time/TimeValues") is None:
                base_iterative_data_node = CGL.newBaseIterativeData(
                    base_node, "Time", 1
                )
                time_values_node = CGU.newNode(
                    "TimeValues", None, [], CGK.DataArray_ts, base_iterative_data_node
                )
                CGU.setValue(time_values_node, np.array([time]))

        return base_node

    def del_base(self, base: str, time: float) -> CGNSTree:
        """Delete a CGNS base node for a specific time.

        Args:
            base (str): The name of the base node to be deleted.
            time (float): The time step for which to delete the CGNS base node.

        Raises:
            KeyError: There is no CGNS tree in this sample / There is no CGNS tree for the provided time.
            KeyError: If there is no base node with the given base name or time.

        Returns:
            CGNSTree: The tree at the provided time (without the deleted node)
        """
        if not self.data:
            raise KeyError("There is no CGNS tree in this sample.")

        if time not in self.data:
            raise KeyError(f"There is no CGNS tree for time {time}.")

        base_node = self.get_base(base, time)
        mesh_tree = self.data[time]

        if base_node is None:
            raise KeyError(f"There is no base node with name {base} for time {time}.")

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
        time = self.resolve_time(time)

        if self.data and time in self.data and self.data[time] is not None:
            return CGH.get_base_names(
                self.data[time], full_path=full_path, unique=unique
            )
        else:
            return []

    def has_base(self, base: str, time: Optional[float] = None) -> bool:
        """Check if a CGNS tree contains a Base with a given name at a specified time.

        Args:
            base (str): The name of the Base to check for in the CGNS tree.
            time (float, optional): The time at which to check for the Base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            bool: `True` if the CGNS tree has a Base called `base`, else return `False`.
        """
        # get_base_names will look for the default time
        return base in self.get_base_names(time=time)

    def has_globals(self, time: Optional[float] = None) -> bool:
        """Check if a CGNS tree contains globals a given name at a specified time.

        Args:
            time (float, optional): The time at which to check for the Base. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            bool: `True` if the CGNS tree has a Base called `Globals`, else return `False`.
        """
        return "Global" in self.get_base_names(time=time)

    def get_base(
        self, base: Optional[str] = None, time: Optional[float] = None
    ) -> CGNSNode | None:
        """Return Base node named `base`.

        If `base` is not specified, checks that there is **at most** one base, else raises an error.

        Args:
            base (str, optional): The name of the Base node to retrieve. Defaults to None. Defaults to None.
            time (float, optional): Time at which you want to retrieve the Base node. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            CGNSNode or None: The Base node with the specified name or None if it is not found.
        """
        time = self.resolve_time(time)
        if time not in self.data or self.data[time] is None:
            logger.warning(f"No mesh exists in the sample at {time=}")
            return None

        if base != "Global":
            base = self.resolve_base(base, time)
        return CGU.getNodeByPath(self.data[time], f"/CGNSTree/{base}")

    # -------------------------------------------------------------------------#
    def init_zone(
        self,
        zone_shape: Array,
        zone_type: str = CGK.Unstructured_s,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> CGNSNode:
        """Initialize a new zone within a CGNS base.

        Args:
            zone_shape (Array): An array specifying the shape or dimensions of the zone.
            zone_type (str, optional): The type of the zone. Defaults to CGK.Unstructured_s.
            zone (str, optional): The name of the zone to initialize. If not provided, uses `mesh_zone_name` specified in Sample initialization. Defaults to None.
            base (str, optional): The name of the base to which the zone will be added. If not provided, the zone will be added to the currently active base. Defaults to None.
            time (float, optional): The time at which to initialize the zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Raises:
            KeyError: If the specified base does not exist. You can create a base using `Sample.init_base(base)`.

        Returns:
            CGLNode: The newly initialized zone node within the CGNS tree.
        """
        _check_names([zone])

        # init_tree will look for default time
        self.init_tree(time)
        # get_base will look for default base and time
        base_node = self.get_base(base, time)
        if base_node is None:
            raise KeyError(
                f"there is no base <{base}>, you should first create one with `Sample.init_base({base=})`"
            )

        zone = self.resolve_zone(zone, base, time)
        if zone is None:
            zone = "Zone"
        zone_node = CGL.newZone(base_node, zone, zone_shape, zone_type)
        return zone_node

    def del_zone(self, zone: str, base: str, time: float) -> CGNSTree:
        """Delete a zone within a CGNS base.

        Args:
            zone (str): The name of the zone to be deleted.
            base (str, optional): The name of the base from which the zone will be deleted. If not provided, the zone will be deleted from the currently active base. Defaults to None.
            time (float, optional): The time step for which to delete the zone. Defaults to None.

        Raises:
            KeyError: There is no CGNS tree in this sample / There is no CGNS tree for the provided time.
            KeyError: If there is no base node with the given base name or time.

        Returns:
            CGNSTree: The tree at the provided time (without the deleted node)
        """
        if self.data is None:  # pragma: no cover
            raise KeyError("There is no CGNS tree in this sample.")

        if time not in self.data:
            raise KeyError(f"There is no CGNS tree for time {time}.")

        zone_node = self.get_zone(zone=zone, base=base, time=time)
        mesh_tree = self.data[time]

        if zone_node is None:
            raise KeyError(
                f"There is no zone node with name {zone} or base node with name {base}."
            )

        return CGU.nodeDelete(mesh_tree, zone_node)

    def get_zone_names(
        self,
        base: Optional[str] = None,
        full_path: bool = False,
        unique: bool = False,
        time: Optional[float] = None,
    ) -> list[str]:
        """Return list of Zone names in Base named `base` with specific time.

        Args:
            base (str, optional): Name of Base where to search Zones. If not specified, checks if there is at most one Base. Defaults to None.
            full_path (bool, optional): If True, returns full paths instead of only Zone names. Defaults to False.
            unique (bool, optional): If True, returns unique names instead of potentially duplicated names. Defaults to False.
            time (float, optional): The time at which to check for the Zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            list[str]: List of Zone names in Base named `base`, empty if there is none or if the Base doesn't exist.
        """
        zone_paths = []

        # get_base will look for default base and time
        base_node = self.get_base(base, time)
        if base_node is not None:
            z_paths = CGU.getPathsByTypeSet(base_node, "CGNSZone_t")
            for pth in z_paths:
                s_pth = pth.split("/")
                assert len(s_pth) == 2
                assert s_pth[0] == base or base is None
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
        zone: str,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> bool:
        """Check if the CGNS tree contains a Zone with the specified name within a specific Base and time.

        Args:
            zone (str): The name of the Zone to check for.
            base (str, optional): The name of the Base where the Zone should be located. If not provided, the function checks all bases. Defaults to None.
            time (float, optional): The time at which to check for the Zone. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            bool: `True` if the CGNS tree has a Zone called `zone` in a Base called `base`, else return `False`.
        """
        # get_zone_names will look for default base and time
        return zone in self.get_zone_names(base, time=time)

    def get_zone(
        self,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> CGNSNode | None:
        """Retrieve a CGNS Zone node by its name within a specific Base and time.

        Args:
            zone (str, optional): The name of the Zone node to retrieve. If not specified, checks that there is **at most** one zone in the base, else raises an error. Defaults to None.
            base (str, optional): The Base in which to seek to zone retrieve. If not specified, checks that there is **at most** one base, else raises an error. Defaults to None.
            time (float, optional): Time at which you want to retrieve the Zone node.

        Returns:
            CGNSNode: Returns a CGNS Zone node if found; otherwise, returns None.
        """
        # get_base will look for default base and time
        base_node = self.get_base(base, time)
        if base_node is None:
            # logger.warning(f"No base with name {base} in this tree")
            return None

        # _zone_attribution will look for default base
        zone = self.resolve_zone(zone, base, time)
        if zone is None:
            # logger.warning(f"No zone with name {zone} in this base ({base})")
            return None

        return CGU.getNodeByPath(base_node, zone)

    def get_zone_type(
        self,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> str:
        """Get the type of a specific zone at a specified time.

        Args:
            zone (str, optional): The name of the zone whose type you want to retrieve. Default is None.
            base (str, optional): The name of the base in which the zone is located. Default is None.
            time (float, optional): The timestamp for which you want to retrieve the zone type. Default is 0.0.

        Raises:
            KeyError: Raised when the specified zone or base does not exist. You should first create the base/zone using `Sample.init_zone(zone, base)`.

        Returns:
            str: The type of the specified zone as a string.
        """
        # get_zone will look for default base, zone and time
        zone_node = self.get_zone(zone=zone, base=base, time=time)

        if zone_node is None:
            raise KeyError(
                f"there is no base/zone <{base}/{zone}>, you should first create one with `Sample.init_zone({zone=},{base=})`"
            )
        return CGU.getValueByPath(zone_node, "ZoneType").tobytes().decode()

    # -------------------------------------------------------------------------#
    def get_nodal_tags(
        self,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> dict[str, Array]:
        """Get the nodal tags for a specified base and zone at a given time.

        Args:
            zone (str, optional): The name of the zone for which element connectivity data is requested. Defaults to None, indicating the default zone.
            base (str, optional): The name of the base for which element connectivity data is requested. Defaults to None, indicating the default base.
            time (float, optional): The time at which element connectivity data is requested. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            dict[str, Array]: A dictionary where keys are nodal tags names and values are NumPy arrays containing the corresponding tag indices.
            The NumPy arrays have shape (num_nodal_tags).
        """
        # get_zone will look for default base, zone and time
        zone_node = self.get_zone(zone=zone, base=base, time=time)

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

    def get_element_tags(
        self,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> dict[str, Array]:
        """Get the element tags for a specified base and zone at a given time.

        Args:
            zone (str, optional): The name of the zone for which element tags are
                requested. Defaults to None.
            base (str, optional): The name of the base for which element tags are
                requested. Defaults to None.
            time (float, optional): The time at which element tags are requested.
                Defaults to None.

        Returns:
            dict[str, Array]: A dictionary where keys are element tag names and
                values are NumPy arrays containing the corresponding element
                indices (0-based).
        """
        zone_node = self.get_zone(zone=zone, base=base, time=time)

        if zone_node is None:
            return {}

        element_tags = {}

        BCPaths = CGU.getPathsByTypeList(zone_node, ["Zone_t", "ZoneBC_t", "BC_t"])
        for BCPath in BCPaths:
            print(BCPath)
            BCNode = CGU.getNodeByPath(zone_node, BCPath)
            BCName = BCNode[0]
            indices = _read_index(BCNode, [1])
            if len(indices) == 0:  # pragma: no cover
                continue

            gl = CGU.getPathsByTypeSet(BCNode, ["GridLocation_t"])
            if gl:
                location = CGU.getValueAsString(CGU.getNodeByPath(BCNode, gl[0]))
            else:  # pragma: no cover
                location = "Vertex"
            if location in ["CellCenter", "FaceCenter"]:
                element_tags[BCName] = indices - 1

        # we treat FamilyName_t and AdditionalFamilyName_t as tag over the topological dimension elements
        fnpath = CGU.getPathsByTypeList(zone_node, ["Zone_t", "FamilyName_t"])
        afnpath = CGU.getPathsByTypeList(
            zone_node, ["Zone_t", "AdditionalFamilyName_t"]
        )
        topo_dim = self.get_topological_dim()

        if topo_dim == 3:
            bulk_elems = CGK.ElementType3D
        elif topo_dim == 2:  # pragma: no cover
            bulk_elems = CGK.ElementType2D
        elif topo_dim == 1:  # pragma: no cover
            bulk_elems = CGK.ElementType1D
        else:  # pragma: no cover
            bulk_elems = CGK.ElementType1D

        if len(fnpath + afnpath):
            self.show_tree()
            bulk_ids = []
            elem_paths = CGU.getAllNodesByTypeSet(zone_node, ["Elements_t"])
            for elem in elem_paths:
                elem_node = CGU.getNodeByPath(zone_node, elem)
                if CGK.ElementType[elem_node[0].lstrip("Elements_")] in bulk_elems:
                    erange = CGU.getValueByPath(elem_node, "ElementRange")
                    bulk_ids += list(range(erange[0] - 1, erange[1] - 1))
            if len(bulk_ids):
                for fpath in fnpath + afnpath:
                    fn = CGU.getNodeByPath(zone_node, fpath)
                    if fn:
                        familyName = CGU.getValueAsString(fn)
                        element_tags[familyName] = bulk_ids

        sorted_element_tags = {
            key: np.sort(value) for key, value in element_tags.items()
        }

        return sorted_element_tags

    # -------------------------------------------------------------------------#
    def get_global(
        self,
        name: str,
        time: Optional[float] = None,
    ) -> Optional[ScalarOrArrayOrStr]:
        """Retrieve a global array by name at a specified time.

        Args:
            name (str): The name of the global array to retrieve.
            time (float, optional): The time step for which to retrieve the global array. If not provided, uses the default time.

        Returns:
            Optional[Array]: The global array if found, otherwise None. Returns a scalar if the array has size 1.
        """
        time = self.resolve_time(time)

        if not self.has_globals(time):
            return None

        global_ = CGU.getValueByPath(self.data[time], "Global/" + name)
        if global_ is None:
            return None

        if getattr(global_, "size", None) == 1:
            return global_[0]

        return global_

    def add_global(
        self,
        name: str,
        global_array: ScalarOrArrayOrStr,
        time: Optional[float] = None,
    ) -> None:
        """Add or update a global array at a specified time.

        Args:
            name (str): The name of the global array to add or update.
            global_array (Array): The array to store.
            time (float, optional): The time step for which to add the global array. If not provided, uses the default time.

        Note:
            If the "Global" base does not exist, it will be created.
            If an array with the same name exists, its value will be updated.
        """
        _check_names(name)
        base_names = self.get_base_names(time=time)
        if "Global" in base_names:
            base_node = self.get_base("Global", time=time)
        else:
            base_node = self.init_base(1, 1, "Global", time)

        if isinstance(global_array, str):  # pragma: no cover
            global_array = np.frombuffer(
                global_array.encode("ascii"), dtype="S1", count=len(global_array)
            )

        if CGU.getValueByPath(base_node, name) is None:
            CGL.newDataArray(base_node, name, value=global_array)
        else:
            global_node = CGU.getNodeByPath(base_node, f"{name}")
            CGU.setValue(global_node, np.asfortranarray(global_array))

    def del_global(
        self,
        name: str,
        time: Optional[float] = None,
    ) -> Array:
        """Delete a global array by name at a specified time.

        Args:
            name (str): The name of the global array to delete.
            time (float, optional): The time step for which to delete the global array. If not provided, uses the default time.

        Raises:
            KeyError: If the global array does not exist at the specified time.

        Returns:
            Array: The value of the deleted global array.
        """
        val = self.get_global(name, time)
        if val is None:
            raise KeyError(
                f"There is no global with name {name} at the specified time."
            )

        base_node = self.get_base("Global", time=time)
        CGU.nodeDelete(base_node, name)

        return val

    def get_global_names(self, time: Optional[float] = None) -> list[str]:
        """Return a list of all global array names at the specified time(s).

        Args:
            time (float, optional): The time step for which to retrieve global names. If not provided, returns names for all available times.

        Returns:
            list[str]: List of global array names (excluding "Time" arrays).
        """
        if time is None:
            all_times = self.get_all_time_values()
        else:
            all_times = [time]
        global_names = []
        for time in all_times:
            base_names = self.get_base_names(time=time)
            if "Global" in base_names:
                base_node = self.get_base("Global", time=time)
                if base_node is not None:
                    global_paths = CGU.getAllNodesByTypeSet(base_node, ["DataArray_t"])
                    for path in global_paths:
                        if "Time" not in path:
                            global_names.append(CGU.getNodeByPath(base_node, path)[0])
        return global_names

    # -------------------------------------------------------------------------#
    def get_nodes(
        self,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
        name: Optional[str] = None,
    ) -> Optional[Array]:
        """Get grid node coordinates from a specified base, zone, and time.

        Args:
            zone (str, optional): The name of the zone to search for. Defaults to None.
            base (str, optional): The name of the base to search for. Defaults to None.
            time (float, optional):  The time value to consider when searching for the zone. If a specific time is not provided, the method will display the tree structure for the default time step.
            name (str, optional): The coordinate array name to retrieve. Supported values are ``CoordinateX``, ``CoordinateY``, and ``CoordinateZ``. If not provided, all coordinates are returned.

        Raises:
            TypeError: Raised if multiple <GridCoordinates> nodes are found. Only one is expected.

        Returns:
            Optional[Array]: A NumPy array containing the grid node coordinates.
            If no matching zone or grid coordinates are found, None is returned.
        """
        # get_zone will look for default base, zone and time
        search_node = self.get_zone(zone=zone, base=base, time=time)

        if search_node is None:
            return None

        grid_paths = CGU.getAllNodesByTypeSet(search_node, ["GridCoordinates_t"])
        if len(grid_paths) == 1:
            grid_node = CGU.getNodeByPath(search_node, grid_paths[0])

            if name == "CoordinateX":
                return CGU.getValueByPath(grid_node, "GridCoordinates/CoordinateX")
            elif name == "CoordinateY":
                return CGU.getValueByPath(grid_node, "GridCoordinates/CoordinateY")
            elif name == "CoordinateZ":
                return CGU.getValueByPath(grid_node, "GridCoordinates/CoordinateZ")
            elif name is not None:
                raise ValueError(f"Unknown coordinate name: {name}")

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

    def set_nodes(
        self,
        nodes: Array,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> None:
        """Set the coordinates of nodes for a specified base and zone at a given time.

        Args:
            nodes (Array): A numpy array containing the new node coordinates.
            zone (str, optional): The name of the zone where the nodes should be updated. Defaults to None.
            base (str, optional): The name of the base where the nodes should be updated. Defaults to None.
            time (float, optional): The time at which the node coordinates should be updated. If a specific time is not provided, the method will display the tree structure for the default time step.

        Raises:
            KeyError: Raised if the specified base or zone do not exist. You should first
                create the base and zone using the `Sample.init_zone(zone,base)` method.
        """
        # get_zone will look for default base, zone and time
        zone_node = self.get_zone(zone=zone, base=base, time=time)

        if zone_node is None:
            raise KeyError(
                f"there is no base/zone <{base}/{zone}>, you should first create one with `Sample.init_zone({zone=},{base=})`"
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

    # -------------------------------------------------------------------------#
    def get_elements(
        self,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> dict[str, Array]:
        """Retrieve element connectivity data for a specified zone, base, and time.

        Args:
            zone (str, optional): The name of the zone for which element
                connectivity data is requested. Defaults to None, indicating the
                default zone.
            base (str, optional): The name of the base for which element
                connectivity data is requested. Defaults to None, indicating the
                default base.
            time (float, optional): The time at which element connectivity data is requested. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            dict[str, Array]: A dictionary where keys are element type names and values are NumPy arrays representing the element connectivity data.
            The NumPy arrays have shape (num_elements, num_nodes_per_element), and element indices are 0-based.
        """
        # get_zone will look for default base, zone and time
        zone_node = self.get_zone(zone=zone, base=base, time=time)

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
        location: str = None,
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> list[str]:
        """Get a set of field names associated with a specified zone, base, location, and/or time.

        For each argument that is not specified, the method will search for fields in all available values for this argument.

        Args:
            location (str, optional): The desired grid location where to search for. Defaults to None.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            zone (str, optional): The name of the zone to search for.
                Defaults to None.
            base (str, optional): The name of the base to search for.
                Defaults to None.
            time (float, optional): The specific time at which to search for. Defaults to None.

        Returns:
            set[str]: A set containing the names of the fields that match the specified criteria.
        """

        def get_field_names_one_time_base_zone_location(
            location: str, zone: str, base: str, time: float
        ) -> list[str]:
            # get_zone will look for default zone, base, time
            search_node = self.get_zone(zone=zone, base=base, time=time)
            if search_node is None:  # pragma: no cover
                return []

            names = []
            # try to find IntegrationPoint on UserDefinedData_t
            solution_paths = CGU.getPathsByTypeSet(
                search_node, [CGK.FlowSolution_t, "UserDefinedData_t"]
            )
            for f_path in solution_paths:
                grid_loc_node = CGU.getValueByPath(
                    search_node, f_path + "/GridLocation"
                )
                if grid_loc_node is not None:
                    if grid_loc_node.tobytes().decode() != location:
                        continue
                else:
                    ##possible an integration point data check Muscat Implementation for details
                    grid_loc_node = CGU.getValueByPath(
                        search_node, f_path + "/gridlocation"
                    )
                    if grid_loc_node is not None:
                        if grid_loc_node.tobytes().decode() != location:
                            continue
                    else:
                        continue

                f_node = CGU.getNodeByPath(search_node, f_path)
                for path in CGU.getPathByTypeFilter(f_node, CGK.DataArray_t):
                    field_name = path.split("/")[-1]
                    if field_name not in [
                        "GridLocation",
                        "ItgRules",
                        "gridlocation",
                        "ItgPointStartOffset",
                        "Ids",
                        "Path",
                    ]:
                        names.append(field_name)

            return names

        field_names = []
        times = [time] if time is not None else self.get_all_time_values()
        for _time in times:
            bases = [base] if base is not None else self.get_base_names(time=_time)
            for _base in bases:
                zones = (
                    [zone]
                    if zone is not None
                    else self.get_zone_names(base=_base, time=_time)
                )
                for _zone in zones:
                    locations = (
                        [location] if location is not None else CGNS_FIELD_LOCATIONS
                    )
                    for _location in locations:
                        field_names += get_field_names_one_time_base_zone_location(
                            location=_location,
                            zone=_zone,
                            base=_base,
                            time=_time,
                        )

        field_names = sorted(set(field_names))

        return field_names

    def get_field(
        self,
        name: str,
        location: str = "Vertex",
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> np.ndarray | None:
        """Retrieve a field with a specified name from a given zone, base, location, and time.

        Args:
            name (str): The name of the field to retrieve.
            location (str, optional): The location at which to retrieve the field. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            zone (str, optional): The name of the zone to search for. Defaults to None.
            base (str, optional): The name of the base to search for. Defaults to None.
            time (float, optional): The time value to consider when searching for the field. If a specific time is not provided, the method will display the tree structure for the default time step.

        Returns:
            Field: A set containing the names of the fields that match the specified criteria.
        """
        # get_zone will look for default time
        search_node = self.get_zone(zone=zone, base=base, time=time)
        if search_node is None:
            return None

        full_field = []
        solution_paths = CGU.getPathsByTypeSet(
            search_node, [CGK.FlowSolution_t, "UserDefinedData_t"]
        )

        for f_path in solution_paths:
            grid_loc = CGU.getValueByPath(search_node, f_path + "/GridLocation")
            if grid_loc is not None:
                if grid_loc.tobytes().decode() != location:
                    continue
            else:
                ##possible an integration point data
                grid_loc = CGU.getValueByPath(search_node, f_path + "/gridlocation")
                if grid_loc is not None:
                    if grid_loc.tobytes().decode() != location:  # pragma: no cover
                        continue
                else:  # pragma: no cover
                    raise

            field = CGU.getValueByPath(search_node, f_path + "/" + name)
            if field is not None and field.size > 0:
                full_field.append(field)

        if not full_field:
            return None
        if len(full_field) == 1:
            return full_field[0]
        raise ValueError(
            f"Multiple fields found with name {name} at location {location}."
        )  # pragma: no cover

    def add_field(
        self,
        name: str,
        field: np.ndarray,
        location: str = "Vertex",
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
        warning_overwrite: bool = True,
    ) -> None:
        """Add a field to a specified zone in the grid.

        Args:
            name (str): The name of the field to be added.
            field (Field): The field data to be added. Integer arrays with dtype
                ``np.int32`` or ``np.int64`` are automatically converted to
                ``np.float64`` (with a warning) for CGNS compatibility.
            location (str, optional): The grid location where the field will be stored. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            zone (str, optional): The name of the zone where the field will be added. Defaults to None.
            base (str, optional): The name of the base where the zone is located. Defaults to None.
            time (float, optional): The time associated with the field. Defaults to 0.
            warning_overwrite (bool, optional): Show warning if a preexisting field is being overwritten. Defaults to True.

        Raises:
            KeyError: Raised if the specified zone does not exist in the given base.
        """
        assert isinstance(field, np.ndarray)
        if field.ndim != 1:
            raise ValueError(
                f"field has {field.ndim} dimensions, but must be a 1D array."
            )
        _check_names([name])
        # init_tree will look for default time
        self.init_tree(time)
        # get_zone will look for default zone, base and time
        zone_node = self.get_zone(zone=zone, base=base, time=time)

        if zone_node is None:
            raise KeyError(
                f"there is no Zone with name {zone} in base {base}. Did you check topological and physical dimensions ?"
            )

        # Check field size consistency with its geometrical support
        n_nodes, n_elems, _ = zone_node[1][0]
        if location == "Vertex" and field.shape[0] != n_nodes:
            raise ValueError(
                f"field has {field.shape[0]} nodes but zone has {n_nodes} nodes (based on the zone node metadata)"
            )
        elif location == "CellCenter" and field.shape[0] != n_elems:
            raise ValueError(
                f"field has {field.shape[0]} nodes but zone has {n_elems} elements (based on the zone node metadata)"
            )

        if field.dtype in (np.int32, np.int64):
            logger.warning(
                f"(add_field) provided field is of type {field.dtype} and has been converted to np.float64 for CGNS compatibility."
            )
            field = field.astype(np.float64)

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
        location: str = "Vertex",
        zone: Optional[str] = None,
        base: Optional[str] = None,
        time: Optional[float] = None,
    ) -> CGNSTree:
        """Delete a field with specified name in the mesh.

        Args:
            name (str): The name of the field to be deleted.
            location (str, optional): The grid location where the field is stored. Defaults to 'Vertex'.
                Possible values : :py:const:`plaid.constants.CGNS_FIELD_LOCATIONS`
            zone (str, optional): The name of the zone from which the field will be deleted. Defaults to None.
            base (str, optional): The name of the base where the zone is located. Defaults to None.
            time (float, optional): The time associated with the field. Defaults to None.

        Raises:
            KeyError: Raised if the specified zone or field does not exist in the given base.

        Returns:
            CGNSTree: The tree at the provided time (without the deleted node)
        """
        # get_zone will look for default zone, base, and time
        zone_node = self.get_zone(zone=zone, base=base, time=time)
        time = self.resolve_time(time)
        mesh_tree = self.data[time]

        if zone_node is None:
            raise KeyError(f"There is no Zone with name {zone} in base {base}.")

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
        time = self.resolve_time(time)

        if self.data is not None:
            CGH.show_cgns_tree(self.data.get(time))
