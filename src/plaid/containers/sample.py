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
from pathlib import Path
from typing import Any, Optional, Union

import CGNS.MAP as CGM
import CGNS.PAT.cgnslib as CGL
import CGNS.PAT.cgnsutils as CGU
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pydantic import Field as PydanticField

from plaid.constants import (
    AUTHORIZED_FEATURE_INFOS,
    AUTHORIZED_FEATURE_TYPES,
    CGNS_FIELD_LOCATIONS,
)
from plaid.containers.features import SampleFeatures
from plaid.containers.utils import get_feature_type_and_details_from
from plaid.types import (
    CGNSTree,
    Feature,
    FeatureIdentifier,
    Scalar,
)
from plaid.utils import cgns_helper as CGH
from plaid.utils.base import delegate, safe_len
from plaid.utils.deprecation import deprecated

logger = logging.getLogger(__name__)


FEATURES_METHODS = [
    "set_default_base",
    "set_default_zone_base",
    "set_default_time",
    "get_all_mesh_times",
    "get_mesh",
    "get_base_names",
    "get_zone_names",
    "get_nodal_tags",
    "get_global",
    "get_global_names",
    "get_nodes",
    "get_elements",
    "get_field_names",
    "get_field",
    "show_tree",
    "set_nodes",
    "del_field",
    "add_field",
    "show_tree",
    "init_base",
    "init_zone",
]


@delegate("features", FEATURES_METHODS)
class Sample(BaseModel):
    """Represents a single sample. It contains data and information related to a single observation or measurement within a dataset.

    By default, the sample is empty but:
        - You can provide a path to a folder containing the sample data, and it will be loaded during initialization.
        - You can provide `SampleFeatures` and `SampleFeatures` instances to initialize the sample with existing data.
        - You can also provide a dictionary of time series data.

    The default `SampleFeatures` instance is initialized with:
        - `data=None`, `links=None`, and `paths=None` (i.e., no mesh data).
        - `mesh_base_name="Base"` and `mesh_zone_name="Zone"`.

    The default `SampleFeatures` instance is initialized with:
        - `scalars=None` (i.e., no scalar data).
    """

    # Pydantic configuration
    model_config = ConfigDict(
        arbitrary_types_allowed=True, revalidate_instances="always", extra="forbid"
    )

    # Attributes
    path: Optional[Union[str, Path]] = PydanticField(
        None,
        description="Path to the folder containing the sample data. If provided, the sample will be loaded from this path during initialization. Defaults to None.",
    )

    features: Optional[SampleFeatures] = PydanticField(
        default_factory=lambda _: SampleFeatures(
            data=None,
            mesh_base_name="Base",
            mesh_zone_name="Zone",
            links=None,
            paths=None,
        ),
        description="An instance of SampleFeatures containing mesh data. Defaults to an empty `SampleFeatures` object.",
    )

    # Private attributes
    _extra_data: Optional[dict] = PrivateAttr(default=None)

    def model_post_init(self, _context: Any) -> None:
        """Post-initialization processing for the Sample model."""
        # Load if path is provided
        if self.path is not None:
            path = Path(self.path)
            self.load(path)

    def copy(self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create a deep copy of the current `Sample` instance.

        Usage of `model_copy(deep=True)` from Pydantic to ensure all internal data is deeply copied.

        Returns:
            A new `Sample` instance with all internal data (scalars, time series, fields, meshes, etc.)
            deeply copied to ensure full isolation from the original.

        Note:
            This operation may be memory-intensive for large samples.
        """
        return self.model_copy(deep=True)

    def get_scalar(self, name: str) -> Optional[Scalar]:
        """Retrieve a scalar value associated with the given name.

        Args:
            name (str): The name of the scalar value to retrieve.

        Returns:
            Scalar or None: The scalar value associated with the given name, or None if the name is not found.
        """
        return self.features.get_global(name)

    def add_scalar(self, name: str, value: Scalar) -> None:
        """Add a scalar value to a dictionary.

        Args:
            name (str): The name of the scalar value.
            value (Scalar): The scalar value to add or update in the dictionary.
        """
        self.features.add_global(name, value)

    def del_scalar(self, name: str) -> Scalar:
        """Delete a scalar value from the dictionary.

        Args:
            name (str): The name of the scalar value to be deleted.

        Raises:
            KeyError: Raised when there is no scalar / there is no scalar with the provided name.

        Returns:
            Scalar: The value of the deleted scalar.
        """
        return self.features.del_global(name)

    def get_scalar_names(self) -> list[str]:
        """Get a set of scalar names available in the object.

        Returns:
            list[str]: A set containing the names of the available scalars.
        """
        return self.features.get_global_names()

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
            path_linked_sample (Union[str,Path]): The absolute path of the folder containing the linked CGNS
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

        if linked_time not in linked_sample.features.data:  # pragma: no cover
            raise KeyError(
                f"There is no CGNS tree for time {linked_time} in linked_sample."
            )
        if time in self.features.data:  # pragma: no cover
            raise KeyError(f"A CGNS tree is already linked in self for time {time}.")

        tree = CGL.newCGNSTree()

        base_names = linked_sample.features.get_base_names(time=linked_time)

        for bn in base_names:
            base_node = linked_sample.features.get_base(bn, time=linked_time)
            base = [bn, base_node[1], [], "CGNSBase_t"]
            tree[2].append(base)

            family = [
                "Bulk",
                np.array([b"B", b"u", b"l", b"k"], dtype="|S1"),
                [],
                "FamilyName_t",
            ]  # maybe get this from linked_sample as well ?
            base[2].append(family)

            zone_names = linked_sample.features.get_zone_names(bn, time=linked_time)
            for zn in zone_names:
                zone_node = linked_sample.features.get_zone(
                    zone_name=zn, base_name=bn, time=linked_time
                )
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
            Types_t = CGU.getAllNodesByTypeSet(sample.features.get_mesh(time), Type_t)
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

        self.features.add_tree(tree, time=time)

        dname = path_linked_sample.parent
        bname = path_linked_sample.name
        self.features._links[time] = [
            [str(dname), bname, fp, fp] for fp in feature_paths
        ]

        return tree

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
                    location=feat_id["location"],
                    zone_name=feat_id["zone_name"],
                    base_name=feat_id["base_name"],
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
        for t in self.features.get_all_mesh_times():
            for bn in self.features.get_base_names(time=t):
                for zn in self.features.get_zone_names(base_name=bn, time=t):
                    if (
                        self.features.get_nodes(base_name=bn, zone_name=zn, time=t)
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
                        for fn in self.features.get_field_names(
                            location=loc, zone_name=zn, base_name=bn, time=t
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
            - `name` is required for scalar and field features.
            - The order of the details must be respected. One cannot specify a detail in the feature_string_identifier string without specified the previous ones.
        """
        splitted_identifier = feature_string_identifier.split("::")

        feature_type = splitted_identifier[0]
        feature_details = [detail for detail in splitted_identifier[1].split("/")]

        assert feature_type in AUTHORIZED_FEATURE_TYPES, "feature_type not known"

        arg_names = AUTHORIZED_FEATURE_INFOS[feature_type]
        assert len(arg_names) >= len(feature_details), "Too much details provided"

        if feature_type == "scalar":
            val = self.get_scalar(feature_details[0])
            if val is None:
                raise KeyError(
                    f"Unknown scalar {feature_details[0]}"
                )  # pragma: no cover
            return val
        elif feature_type == "field":
            kwargs = {arg_names[i]: detail for i, detail in enumerate(feature_details)}
            for k in kwargs:
                if kwargs[k] == "":
                    kwargs[k] = None
            if "time" in kwargs:
                kwargs["time"] = float(kwargs["time"])
            return self.get_field(**kwargs)
        elif feature_type == "nodes":
            kwargs = {arg_names[i]: detail for i, detail in enumerate(feature_details)}
            for k in kwargs:
                if kwargs[k] == "":
                    kwargs[k] = None
            if "time" in kwargs:
                kwargs["time"] = float(kwargs["time"])
            return self.get_nodes(**kwargs).flatten()

    def get_feature_from_identifier(
        self, feature_identifier: FeatureIdentifier
    ) -> Feature:
        """Retrieve a feature object based on a structured identifier dictionary.

        The `feature_identifier` must include a `"type"` key specifying the feature kind:
            - `"scalar"`       → calls `scalars.get(name)`
            - `"field"`        → calls `get_field(name, base_name, zone_name, location, time)`
            - `"nodes"`        → calls `get_nodes(base_name, zone_name, time)`

        Required keys:
            - `"type"`: one of `"scalar"`, `"field"`, or `"nodes"`
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
            - `"field"`        → calls `get_field(name, base_name, zone_name, location, time)`
            - `"nodes"`        → calls `get_nodes(base_name, zone_name, time)`

        Required keys:
            - `"type"`: one of `"scalar"`, `"field"`, or `"nodes"`
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

        This method applies updates to scalars, fields, or nodes
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
        elif feature_type == "field":
            self.add_field(**feature_details, field=feature, warning_overwrite=False)
        elif feature_type == "nodes":
            physical_dim_arg = {
                k: v for k, v in feature_details.items() if k in ["base_name", "time"]
            }
            phys_dim = self.features.get_physical_dim(**physical_dim_arg)
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
            feature_identifiers (FeatureIdentifier or list of FeatureIdentifier): One or more feature identifiers.
            features (Feature or list of Feature): One or more features corresponding
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
            feature_identifiers[i_id] = FeatureIdentifier(feat_id)

        sample = self if in_place else self.copy()

        for feat_id, feat in zip(feature_identifiers, features):
            sample._add_feature(feat_id, feat)

        return sample

    def extract_sample_from_identifier(
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

        source_sample = self.copy()
        source_sample.del_all_fields()

        sample = Sample()

        for feat_id in feature_identifiers:
            feature = self.get_feature_from_identifier(feat_id)

            if feature is not None:
                # get time of current feature
                time = self.features.get_time_assignment(time=feat_id.get("time"))

                # if the constructed sample does not have a tree, add the one from the source sample, with no field
                if len(sample.features.get_base_names(time=time, globals=False)) == 0:
                    sample.features.add_tree(source_sample.features.get_mesh(time))
                    for name in sample.features.get_global_names(time=time):
                        sample.features.del_global(name, time)

                sample._add_feature(feat_id, feature)

        sample._extra_data = copy.deepcopy(self._extra_data)

        return sample

    @deprecated(
        "Use extract_sample_from_identifier() instead",
        version="0.1.8",
        removal="0.2",
    )
    def from_features_identifier(
        self,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
    ) -> Self:
        """DEPRECATED: Use extract_sample_from_identifier() instead."""
        return self.extract_sample_from_identifier(
            feature_identifiers
        )  # pragma: no cover

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

        # DELETE LATER IF CONFIRMED THIS IS NOT NEEDED (WITH GLOBAL, THERE IS ALWAYS A TREE)
        # for feat_id in all_features_identifiers:
        #     # if trying to add a field or nodes, must check if the corresponding tree exists, and add it if not
        #     if feat_id["type"] in ["field", "nodes"]:
        #         # get time of current feature
        #         time = sample.features.get_time_assignment(time=feat_id.get("time"))

        #         # if the constructed sample does not have a tree, add the one from the source sample, with no field
        #         if not merged_dataset.features.get_mesh(time):
        #             merged_dataset.features.add_tree(source_sample.get_mesh(time))

        return merged_dataset.update_features_from_identifier(
            feature_identifiers=all_features_identifiers,
            features=all_features,
            in_place=in_place,
        )

    # -------------------------------------------------------------------------#
    def save(self, path: Union[str, Path], overwrite: bool = False) -> None:
        """Save the Sample in directory `path`.

        Args:
            path (Union[str,Path]): relative or absolute directory path.
            overwrite (bool): target directory overwritten if True.
        """
        path = Path(path)

        if path.is_dir():
            if overwrite:
                shutil.rmtree(path)
                logger.warning(f"Existing {path} directory has been reset.")
            elif len(list(path.glob("*"))):
                raise ValueError(
                    f"directory {path} already exists and is not empty. Set `overwrite` to True if needed."
                )

        path.mkdir(exist_ok=True)

        mesh_dir = path / "meshes"

        if self.features.data:
            mesh_dir.mkdir()
            for i, time in enumerate(self.features.data.keys()):
                outfname = mesh_dir / f"mesh_{i:09d}.cgns"
                status = CGM.save(
                    str(outfname),
                    self.features.data[time],
                    links=self.features._links.get(time),
                )
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
            It calls 'load' function during execution.
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
            # self.features = {}
            self.features._links = {}
            self.features._paths = {}
            for i in range(nb_meshes):
                tree, links, paths = CGM.load(str(meshes_dir / f"mesh_{i:09d}.cgns"))
                time = CGH.get_time_values(tree)

                (
                    self.features.data[time],
                    self.features._links[time],
                    self.features._paths[time],
                ) = (
                    tree,
                    links,
                    paths,
                )
                for i in range(len(self.features._links[time])):  # pragma: no cover
                    self.features._links[time][i][0] = str(
                        meshes_dir / self.features._links[time][i][0]
                    )

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

        # fields
        times = self.features.get_all_mesh_times()
        nb_timestamps = len(times)
        str_repr += f"{nb_timestamps} timestamp{'' if nb_timestamps == 1 else 's'}, "

        field_names = set()
        for time in times:
            ## Need to include all possible location within the count
            base_names = self.features.get_base_names(time=time)
            for bn in base_names:
                zone_names = self.features.get_zone_names(base_name=bn)
                for zn in zone_names:
                    for location in CGNS_FIELD_LOCATIONS:
                        field_names = field_names.union(
                            self.features.get_field_names(
                                location=location, zone_name=zn, base_name=bn, time=time
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
        scalar_names = self.get_scalar_names()
        if scalar_names:
            summary += f"Scalars ({len(scalar_names)}):\n"
            for name in scalar_names:
                value = self.get_scalar(name)
                summary += f"  - {name}: {value}\n"
            summary += "\n"

        # Mesh information
        times = self.features.get_all_mesh_times()
        summary += f"Meshes ({len(times)} timestamps):\n"
        if times:
            for time in times:
                summary += f"    Time: {time}\n"
                base_names = self.features.get_base_names(time=time)
                for base_name in base_names:
                    summary += f"        Base: {base_name}\n"
                    zone_names = self.features.get_zone_names(
                        base_name=base_name, time=time
                    )
                    for zone_name in zone_names:
                        summary += f"            Zone: {zone_name}\n"
                        # Nodes, nodal tags and fields at verticies
                        nodes = self.get_nodes(
                            zone_name=zone_name, base_name=base_name, time=time
                        )
                        if nodes is not None:
                            nb_nodes = nodes.shape[0]
                            nodal_tags = self.features.get_nodal_tags(
                                zone_name=zone_name, base_name=base_name, time=time
                            )
                            summary += f"                Nodes ({nb_nodes})\n"
                            if len(nodal_tags) > 0:
                                summary += f"                Tags ({len(nodal_tags)}): {', '.join([f'{k} ({len(v)})' for k, v in nodal_tags.items()])}\n"

                        for location in CGNS_FIELD_LOCATIONS:
                            field_names = self.get_field_names(
                                location=location,
                                zone_name=zone_name,
                                base_name=base_name,
                                time=time,
                            )
                            if field_names:
                                summary += f"                Location: {location}\n                    Fields ({len(field_names)}): {', '.join(field_names)}\n"

                        # Elements and fields at elements
                        elements = self.features.get_elements(
                            zone_name=zone_name, base_name=base_name, time=time
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
                Has time series: False
                Has meshes: True
                Total unique fields: 8
                Field names: M_iso, mach, nut, ro, roe, rou, rov, sdf
        """
        report = "Sample Completeness Check:\n"
        report += "=" * 30 + "\n"

        # Check if sample has basic features
        has_scalars = len(self.get_scalar_names()) > 0
        has_meshes = len(self.features.get_all_mesh_times()) > 0

        report += f"Has scalars: {has_scalars}\n"
        report += f"Has meshes: {has_meshes}\n"

        if has_meshes:
            times = self.features.get_all_mesh_times()
            total_fields = set()
            for time in times:
                base_names = self.features.get_base_names(time=time)
                for base_name in base_names:
                    zone_names = self.features.get_zone_names(
                        base_name=base_name, time=time
                    )
                    for zone_name in zone_names:
                        for location in CGNS_FIELD_LOCATIONS:
                            field_names = self.get_field_names(
                                location=location,
                                zone_name=zone_name,
                                base_name=base_name,
                                time=time,
                            )
                            total_fields.update(field_names)

            report += f"Total unique fields: {len(total_fields)}\n"
            if total_fields:
                report += f"Field names: {', '.join(sorted(total_fields))}\n"

        return report
