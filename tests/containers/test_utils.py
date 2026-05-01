# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from pathlib import Path

import numpy as np
import pytest

from plaid.containers.utils import (
    decode_cgns_url_details,
    get_feature_details_from_path,
    get_number_of_samples,
    get_sample_ids,
    validate_required_infos,
)

# %% Fixtures


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


# %% Tests


class Test_Container_Utils:
    def test_get_sample_ids(self, current_directory):
        dataset_path = current_directory / "dataset" /"data"/"test"
        assert get_sample_ids(dataset_path) == list(np.arange(0, 10))

    def test_get_number_of_samples(self, current_directory):
        dataset_path = current_directory / "dataset" /"data"/"test"
        assert get_number_of_samples(dataset_path) == 10

    def test_get_sample_ids_with_str(self, current_directory):
        dataset_path = current_directory / "dataset" /"data"/"test"
        assert get_sample_ids(str(dataset_path)) == list(np.arange(0, 10))

    def test_get_number_of_samples_with_str(self, current_directory):
        dataset_path = current_directory / "dataset" /"data"/"test"
        assert get_number_of_samples(str(dataset_path)) == 10

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (
                "CGNSLibraryVersion",
                {"type": "cgns", "sub_type": "library_version"},
            ),
            ("Global", {"type": "global", "sub_type": "root"}),
            (
                "Global/Time/IterationValues",
                {
                    "type": "global",
                    "sub_type": "time",
                    "name": "IterationValues",
                },
            ),
            (
                "Global/Mach",
                {"type": "global", "sub_type": "scalar", "name": "Mach"},
            ),
            (
                "Global_times/Q",
                {"type": "global", "sub_type": "scalar", "name": "Q"},
            ),
            ("Base_2_2", {"base": "Base_2_2", "type": "base"}),
            (
                "Base_2_2/Zone",
                {"base": "Base_2_2", "zone": "Zone", "type": "zone"},
            ),
            (
                "Base_2_2/Zone/GridCoordinates",
                {
                    "base": "Base_2_2",
                    "zone": "Zone",
                    "type": "coordinate",
                    "sub_type": "node",
                },
            ),
            (
                "Base_2_2/Zone/GridCoordinates/CoordinateX",
                {
                    "base": "Base_2_2",
                    "zone": "Zone",
                    "type": "coordinate",
                    "sub_type": "node",
                    "name": "CoordinateX",
                },
            ),
            (
                "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity",
                {
                    "base": "Base_2_2",
                    "zone": "Zone",
                    "type": "elements",
                    "element_type": "QUAD_4",
                    "sub_type": "connectivity",
                },
            ),
            (
                "Base_2_2/Zone/Elements_QUAD_4/ElementRange",
                {
                    "base": "Base_2_2",
                    "zone": "Zone",
                    "type": "elements",
                    "element_type": "QUAD_4",
                    "sub_type": "range",
                },
            ),
            (
                "Base_2_2/Zone/VertexFields/materialID",
                {
                    "base": "Base_2_2",
                    "zone": "Zone",
                    "type": "field",
                    "location": "Vertex",
                    "name": "materialID",
                },
            ),
            (
                "Base_2_2/Zone/PointData/rov",
                {
                    "base": "Base_2_2",
                    "zone": "Zone",
                    "type": "field",
                    "location": "Vertex",
                    "name": "rov",
                },
            ),
            (
                "Base_2_2/Zone/Time/IterationValues",
                {
                    "base": "Base_2_2",
                    "zone": "Zone",
                    "type": "other",
                    "path": "Base_2_2/Zone/Time/IterationValues",
                },
            ),
        ],
    )
    def test_decode_cgns_url_details(self, url, expected):
        assert decode_cgns_url_details(url) == expected

    def test_decode_cgns_url_details_invalid_base(self):
        with pytest.raises(AssertionError, match="path not recognized"):
            decode_cgns_url_details("Dummy")

    def test_decode_cgns_url_details_zone_bc_current_behavior(self):
        with pytest.raises(NameError):
            decode_cgns_url_details("Base_2_2/Zone/ZoneBC/BottomLeft")

    # def test_check_features_type_homogeneity(self):
    #     check_features_type_homogeneity(
    #         [{"type": "scalar", "name": "Mach"}, {"type": "scalar", "name": "P"}]
    #     )

    # def test_check_features_type_homogeneity_fail_type(self):
    #     with pytest.raises(AssertionError):
    #         check_features_type_homogeneity(0)

    # def test_check_features_type_homogeneity_fail(self):
    #     with pytest.raises(AssertionError):
    #         check_features_type_homogeneity(
    #             [{"type": "scalar", "name": "Mach"}, {"type": "nodes"}]
    #         )

    # def test_has_duplicates_feature_ids(self):
    #     assert not has_duplicates_feature_ids(
    #         [{"type": "scalar", "name": "Mach"}, {"type": "scalar", "name": "P"}]
    #     )
    #     assert has_duplicates_feature_ids(
    #         [{"type": "scalar", "name": "Mach"}, {"type": "scalar", "name": "Mach"}]
    #     )

    # def test_get_feature_details_from_path(self):
    #     details = get_feature_details_from_path("Base_2_2")
    #     assert details["base"] == "Base_2_2"

    #     details = get_feature_details_from_path("Global/toto")
    #     assert details["type"] == "global"
    #     assert details["name"] == "toto"

    #     details = get_feature_details_from_path("Base_2_2/Zone")
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Zone"

    #     details = get_feature_details_from_path(
    #         "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity"
    #     )
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Zone"
    #     assert details["type"] == "elements"
    #     assert details["sub_type"] == "connectivity"
    #     assert details["element_type"] == "QUAD_4"

    #     details = get_feature_details_from_path(
    #         "Base_2_2/Zone/Elements_QUAD_4/ElementRange"
    #     )
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Zone"
    #     assert details["type"] == "elements"
    #     assert details["sub_type"] == "range"
    #     assert details["element_type"] == "QUAD_4"

    #     details = get_feature_details_from_path(
    #         "Base_2_2/Zone/GridCoordinates/CoordinateX"
    #     )
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Zone"
    #     assert details["type"] == "coordinate"
    #     assert details["sub_type"] == "node"
    #     assert details["name"] == "CoordinateX"

    #     details = get_feature_details_from_path("Base_2_2/Zone/VertexFields/materialID")
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Zone"
    #     assert details["type"] == "field"
    #     assert details["location"] == "Vertex"
    #     assert details["name"] == "materialID"

    #     details = get_feature_details_from_path(
    #         "Base_2_2/Zone/ZoneBC/BottomLeft/PointList"
    #     )
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Zone"
    #     assert details["type"] == "boundary_condition"
    #     assert details["sub_type"] == "PointList"
    #     assert details["name"] == "BottomLeft"

    #     details = get_feature_details_from_path("Base_2_2/Time")
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Time"
    #     assert details["type"] == "zone"

    #     details = get_feature_details_from_path("Base_2_2/Time/IterationValues")
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Time"
    #     assert details["type"] == "other"
    #     assert details["path"] == "Base_2_2/Time/IterationValues"

    #     details = get_feature_details_from_path("Base_2_2/Time/TimeValues")
    #     assert details["base"] == "Base_2_2"
    #     assert details["zone"] == "Time"
    #     assert details["type"] == "other"
    #     assert details["path"] == "Base_2_2/Time/TimeValues"

    #     with pytest.raises(AssertionError):
    #         get_feature_details_from_path("Dummy")

    #     with pytest.raises(AssertionError):
    #         get_feature_details_from_path("Dummy/Dummy/Dummy/Dummy/Dummy/Dummy/Dummy")

    # def test_validate_required_infos(self):
    #     infos = {
    #         "legal": {"owner": "Joh Doe", "license": "cc-by-sa-4.0"},
    #     }
    #     validate_required_infos(infos)

    #     infos_missing_license = {
    #         "legal": {
    #             "owner": "Joh Doe",
    #         },
    #     }
    #     with pytest.raises(ValueError):
    #         validate_required_infos(infos_missing_license)

    #     infos_dummy = {"dummy": "toto"}
    #     with pytest.raises(AssertionError):
    #         validate_required_infos(infos_dummy)
