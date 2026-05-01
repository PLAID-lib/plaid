"""Implementation of the `Sample` container."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports
import sys
from ..types.common import ArrayDType
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import copy
import logging
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional, Union

import CGNS.MAP as CGM
import CGNS.PAT.cgnsutils as CGU
import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pydantic import Field as PydanticField

from ..types import Array, ArrayDType
from .utils import get_feature_details_from_path

from ..constants import (
    AUTHORIZED_FEATURE_INFOS,
    AUTHORIZED_FEATURE_TYPES_T,
    AUTHORIZED_FEATURE_TYPES,
    CGNS_FIELD_LOCATIONS,
)

from .features import SampleFeatures
from .utils import get_feature_type_and_details_from
from ..utils import cgns_helper as CGH
from ..utils.base import delegate_methods, safe_len

logger = logging.getLogger(__name__)

CGNS_WORKER = Path(__file__).parent.parent / "utils" / "cgns_worker.py"

FEATURES_METHODS = [
    "set_default_base",
    "set_default_zone_base",
    "resolve_base",
    "resolve_zone",
    "set_default_time",
    "get_all_time_values",
    "get_all_mesh_times",
    "get_tree",
    "get_base_names",
    "get_zone_names",
    "get_nodal_tags",
    "has_globals",
    "get_global",
    "add_global",
    "del_global",
    "get_global_names",
    "get_nodes",
    "get_elements",
    "get_field_names",
    "get_field",
    "show_tree",
    "set_nodes",
    "del_field",
    "add_field",
    "init_base",
    "init_zone",
    "init_tree",
    "add_tree",
    "del_tree",
    "set_trees",
]

@delegate_methods("features", FEATURES_METHODS)
class Sample(BaseModel):
    """Represents a single sample. It contains data and information related to a single observation or measurement within a dataset.

    By default, the sample is empty but:
        - You can provide a path to a folder containing the sample data, and it will be loaded during initialization.
        - You can provide `SampleFeatures` and `SampleFeatures` instances to initialize the sample with existing data.

    The default `SampleFeatures` instance is initialized with `data=None` (i.e., no mesh data).
    """

    # Pydantic configuration
    #TODO(FB) check why arbitrary_types_allowed is needed, and if it can be removed
    model_config = ConfigDict(
        arbitrary_types_allowed=True, revalidate_instances="always", extra="forbid"
    )

    # Attributes
    path: Optional[Union[str, Path]] = PydanticField(
        None,
        description="Path to the folder containing the sample data. If provided, the sample will be loaded from this path during initialization. Defaults to None.",
    )

    features: SampleFeatures = PydanticField(
        default_factory=lambda : SampleFeatures(data=None),
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
            A new `Sample` instance with all internal data (scalars, fields, meshes, etc.)
            deeply copied to ensure full isolation from the original.

        Note:
            This operation may be memory-intensive for large samples.
        """
        return self.model_copy(deep=True)

    def get_all_features_identifiers_by_type(self, feature_type: AUTHORIZED_FEATURE_TYPES_T) -> list[str]:

        """Get all features identifiers of a given type from the sample.

        Args:
            feature_type (str): Type of features to return

        Returns:
            list[FeatureIdentifier]: A list of dictionaries containing the identifiers of a given type of all features in the sample.
        """

        assert feature_type in AUTHORIZED_FEATURE_TYPES, "feature_type not known"
        if feature_type ==  "scalar":
            return self.get_global_names()
        elif feature_type ==  "field":
            return self.get_field_names()
        elif feature_type ==  "nodes":
            return ["Coordinate" + n for _,n in  zip(range(self.features.get_physical_dim()), ["X","Y","Z"]) ]

    def get_feature_by_url(self, path: str, time: Optional[int] = None) -> np.number | np.ndarray | None:
        """Retrieve a feature value from the sample's CGNS mesh using a CGNS-style url.

        Args:
            path (str): CGNS node path relative to the mesh root (for example
                "BaseName/ZoneName/GridCoordinates/CoordinateX" or
                "BaseName/ZoneName/Solution/FieldName").
            time (Optional[int], optional): Time selection for the mesh. If an integer,
                it is interpreted via the SampleFeatures time-assignment logic
                (see SampleFeatures.resolve_time). If None, the default time
                assignment is used. Defaults to None.

        Returns:
            Feature: The value stored at the given CGNS path. This may be a numpy array, a scalar, or None if the node has no value.

        Note:
            - This is a thin wrapper around CGNS.PAT.cgnsutils.getValueByPath and Sample.get_tree(time). Callers should handle a returned None when the path or value does not exist.
            - For field-like features, prefer using Sample.get_field which applies additional validation and selection logic.
        """
        time = self.features.resolve_time(time)
        return CGU.getValueByPath(self.get_tree(time), path)

    # def get_feature_from_details(self, feature_details):
    #     #TODO (FB) dont know if this is useful

    #     my_type = feature_details.pop("type", None)
    #     my_sub_type = feature_details.pop("sub_type", None)
    #     if my_type == 'coordinate':
    #         print(feature_details)
    #         return self.get_nodes(**feature_details).flatten()
    #     elif my_type == 'field':
    #         print(feature_details)
    #         return self.get_field(**feature_details)
    #     elif my_type == 'global':
    #         return self.get_global(**feature_details)

    #     print(feature_details)
    #     print(my_type)
    #     raise RuntimeError(f"Cant find a featurn for the given details")

    def add_feature(
        self,
        feature_url: str,
        feature: ArrayDType,
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
        #feature_type, feature_details = get_feature_type_and_details_from(
        #    feature_identifier
        #)

        from .utils import decode_cgns_url_details
        feature_details = decode_cgns_url_details(feature_url)

        feature_type = feature_details.pop("type")
        feature_subtype = feature_details.pop("sub_type",None)

        if feature_type == "global":
            if safe_len(feature) == 1:
                feature = feature[0]
            self.add_global(**feature_details, global_array=feature)
        elif feature_type == "field":
            self.add_field(**feature_details, field=feature, warning_overwrite=False)
        elif feature_type == "coordinate":
            if feature_details.get("name", None) is not None:   # pragma: no cover
                raise ValueError("Must set the 3 coordinate at the same time")
            physical_dim_arg = {
                k: v for k, v in feature_details.items() if k in ["base_name", "time"]
            }
            phys_dim = self.features.get_physical_dim(**physical_dim_arg)
            self.set_nodes(**feature_details, nodes=feature.reshape((-1, phys_dim)))
        else:   # pragma: no cover
            print(feature_details)
            raise RuntimeError(f"feature_type not allowed : {feature_type}")

        return self

    def del_feature(
        self,
        feature_url: str,
    ) -> Self:
        """Remove a feature from current sample.

        This method applies updates to scalars, time series, fields, or nodes using feature identifiers.

        Args:
            feature_identifier (dict): A feature identifier.

        Returns:
            Self: The updated sample

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """

        from .utils import decode_cgns_url_details
        feature_details = decode_cgns_url_details(feature_url)

        feature_type = feature_details.pop("type")
        feature_subtype = feature_details.pop("sub_type",None)


        if feature_type == "global":
            self.del_global(**feature_details)
        elif feature_type == "field":
            self.del_field(**feature_details)
        elif feature_type == "coordinate":
            raise NotImplementedError("Deleting node features is not implemented.")
        else:   # pragma: no cover
            print(feature_details)
            raise RuntimeError(f"feature_type not allowed : {feature_type}")

        return self

    def update_features_by_url(
        self,
        feature_identifiers: dict[
            int, Union[str, list[str]]
        ],
        features: Union[ArrayDType, list[ArrayDType]],
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

    # def extract_sample_by_url(
    #     self,
    #     feature_identifiers: Union[str, list[str]],
    # ) -> Self:
    #     """Extract features of the sample by their identifier(s) and return a new sample containing these features.

    #     This method applies updates to scalars, fields, or nodes
    #     using feature identifiers

    #     Args:
    #         feature_identifiers (dict or list of dict): One or more feature identifiers.

    #     Returns:
    #         Self: New sample containing the provided feature identifiers

    #     Raises:
    #         AssertionError: If types are inconsistent or identifiers contain unexpected keys.
    #     """
    #     assert isinstance(feature_identifiers, dict) or isinstance(
    #         feature_identifiers, list
    #     ), "Check types of feature_identifiers argument"
    #     if isinstance(feature_identifiers, dict):
    #         feature_identifiers = [feature_identifiers]

    #     source_sample = self.copy()
    #     source_sample.del_all_fields()

    #     sample = Sample()

    #     for feat_id in feature_identifiers:
    #         feature = self.get_feature_from_identifier(feat_id)

    #         if feature is not None:
    #             # get time of current feature
    #             time = self.features.resolve_time(time=feat_id.get("time"))

    #             # if the constructed sample does not have a tree, add the one from the source sample, with no field
    #             if len(sample.features.get_base_names(time=time)) == 0:
    #                 sample.features.add_tree(source_sample.features.get_tree(time))
    #                 for name in sample.features.get_global_names(time=time):
    #                     sample.features.del_global(name, time)

    #             sample.add_feature(feat_id, feature)

    #     sample._extra_data = copy.deepcopy(self._extra_data)

    #     return sample

    # def merge_features(self, sample: Self, in_place: bool = False) -> Self:
    #     """Merge features from another sample into the current sample.

    #     This method applies updates to scalars, fields, or nodes
    #     using features from another sample. When `in_place=False`, a deep copy of the sample is created
    #     before applying updates, ensuring full isolation from the original.

    #     Args:
    #         sample (Sample): The sample from which features will be merged.
    #         in_place (bool, optional): If True, modifies the current sample in place.
    #             If False, returns a deep copy with updated features.

    #     Returns:
    #         Self: The updated sample (either the current instance or a new copy).
    #     """
    #     merged_dataset = self if in_place else self.copy()

    #     all_features_identifiers = sample.get_all_features_identifiers()
    #     all_features = sample.get_features_from_identifiers(all_features_identifiers)

    #     feature_types = set([feat_id["type"] for feat_id in all_features_identifiers])

    #     # if field or node features are to extract, copy the source sample and delete all fields
    #     if "field" in feature_types or "nodes" in feature_types:
    #         source_sample = sample.copy()
    #         source_sample.del_all_fields()

    #     # DELETE LATER IF CONFIRMED THIS IS NOT NEEDED (WITH GLOBAL, THERE IS ALWAYS A TREE)
    #     # for feat_id in all_features_identifiers:
    #     #     # if trying to add a field or nodes, must check if the corresponding tree exists, and add it if not
    #     #     if feat_id["type"] in ["field", "nodes"]:
    #     #         # get time of current feature
    #     #         time = sample.features.resolve_time(time=feat_id.get("time"))

    #     #         # if the constructed sample does not have a tree, add the one from the source sample, with no field
    #     #         if not merged_dataset.features.get_tree(time):
    #     #             merged_dataset.features.add_tree(source_sample.get_tree(time))

    #     return merged_dataset.update_features_by_url(
    #         feature_identifiers=all_features_identifiers,
    #         features=all_features,
    #         in_place=in_place,
    #     )

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

        if self.features.data:
            mesh_dir.mkdir()
            for i, time in enumerate(self.features.data.keys()):
                outfname = mesh_dir / f"mesh_{i:09d}.cgns"
                if memory_safe:
                    tmpfile = mesh_dir / f"mesh_{i:09d}.pkl"
                    with open(tmpfile, "wb") as f:
                        pickle.dump(self.features.data[time], f)

                    cmd = [sys.executable, str(CGNS_WORKER), tmpfile, str(outfname)]
                    subprocess.run(cmd)
                    logger.debug(f"save -> {outfname}")

                else:
                    status = CGM.save(str(outfname), self.features.data[time])
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

                (self.features.data[time],) = (tree,)

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
        times = self.features.get_all_time_values()
        nb_timestamps = len(times)
        str_repr += f"{nb_timestamps} timestamp{'' if nb_timestamps == 1 else 's'}, "

        field_names = set()
        for time in times:
            ## Need to include all possible location within the count
            base_names = self.features.get_base_names(time=time)
            for bn in base_names:
                zone_names = self.features.get_zone_names(base_name=bn, time=time)
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
        scalar_names = self.get_global_names()
        if scalar_names:
            summary += f"Scalars ({len(scalar_names)}):\n"
            for name in scalar_names:
                value = self.get_global(name)
                summary += f"  - {name}: {value}\n"
            summary += "\n"

        # Mesh information
        times = self.features.get_all_time_values()
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
                            zone=zone_name, base=base_name, time=time
                        )
                        if nodes is not None:
                            nb_nodes = nodes.shape[0]
                            nodal_tags = self.features.get_nodal_tags(
                                zone=zone_name, base=base_name, time=time
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
                Has meshes: True
                Total unique fields: 8
                Field names: M_iso, mach, nut, ro, roe, rou, rov, sdf
        """
        report = "Sample Completeness Check:\n"
        report += "=" * 30 + "\n"

        # Check if sample has basic features
        has_scalars = len(self.get_global_names()) > 0
        has_meshes = len(self.features.get_all_time_values()) > 0

        report += f"Has scalars: {has_scalars}\n"
        report += f"Has meshes: {has_meshes}\n"

        if has_meshes:
            times = self.features.get_all_time_values()
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

if TYPE_CHECKING:
    # Inheriting from the Protocol (SampleFeatures)  inside TYPE_CHECKING
    # automatically adds all its methods to Sample's autocomplete.
    class Sample(Sample, SampleFeatures): ...
