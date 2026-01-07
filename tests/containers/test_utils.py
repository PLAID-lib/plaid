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
    check_features_type_homogeneity,
    get_feature_details_from_path,
    get_number_of_samples,
    get_sample_ids,
    has_duplicates_feature_ids,
)

# %% Fixtures


@pytest.fixture()
def current_directory():
    return Path(__file__).absolute().parent


# %% Tests


class Test_Container_Utils:
    def test_get_sample_ids(self, current_directory):
        dataset_path = current_directory / "dataset"
        assert get_sample_ids(dataset_path) == list(np.arange(0, 3))

    def test_get_number_of_samples(self, current_directory):
        dataset_path = current_directory / "dataset"
        assert get_number_of_samples(dataset_path) == 3

    def test_get_sample_ids_with_str(self, current_directory):
        dataset_path = current_directory / "dataset"
        assert get_sample_ids(str(dataset_path)) == list(np.arange(0, 3))

    def test_get_number_of_samples_with_str(self, current_directory):
        dataset_path = current_directory / "dataset"
        assert get_number_of_samples(str(dataset_path)) == 3

    def test_check_features_type_homogeneity(self):
        check_features_type_homogeneity(
            [{"type": "scalar", "name": "Mach"}, {"type": "scalar", "name": "P"}]
        )

    def test_check_features_type_homogeneity_fail_type(self):
        with pytest.raises(AssertionError):
            check_features_type_homogeneity(0)

    def test_check_features_type_homogeneity_fail(self):
        with pytest.raises(AssertionError):
            check_features_type_homogeneity(
                [{"type": "scalar", "name": "Mach"}, {"type": "nodes"}]
            )

    def test_has_duplicates_feature_ids(self):
        assert not has_duplicates_feature_ids(
            [{"type": "scalar", "name": "Mach"}, {"type": "scalar", "name": "P"}]
        )
        assert has_duplicates_feature_ids(
            [{"type": "scalar", "name": "Mach"}, {"type": "scalar", "name": "Mach"}]
        )

    def test_get_feature_details_from_path(self):
        details = get_feature_details_from_path("Base_2_2")
        assert details["base"] == "Base_2_2"

        details = get_feature_details_from_path("Global/toto")
        assert details["type"] == "global"
        assert details["name"] == "toto"

        details = get_feature_details_from_path("Base_2_2/Zone")
        assert details["base"] == "Base_2_2"
        assert details["zone"] == "Zone"

        details = get_feature_details_from_path(
            "Base_2_2/Zone/Elements_QUAD_4/ElementConnectivity"
        )
        assert details["base"] == "Base_2_2"
        assert details["zone"] == "Zone"
        assert details["type"] == "element_connectivity"
        assert details["element"] == "Elements_QUAD_4"

        details = get_feature_details_from_path(
            "Base_2_2/Zone/Elements_QUAD_4/ElementRange"
        )
        assert details["base"] == "Base_2_2"
        assert details["zone"] == "Zone"
        assert details["type"] == "element_range"
        assert details["element"] == "Elements_QUAD_4"

        details = get_feature_details_from_path(
            "Base_2_2/Zone/GridCoordinates/CoordinateX"
        )
        assert details["base"] == "Base_2_2"
        assert details["zone"] == "Zone"
        assert details["type"] == "node_coordinate"
        assert details["name"] == "CoordinateX"

        details = get_feature_details_from_path("Base_2_2/Zone/VertexFields/materialID")
        assert details["base"] == "Base_2_2"
        assert details["zone"] == "Zone"
        assert details["type"] == "field"
        assert details["location"] == "Vertex"
        assert details["name"] == "materialID"

        details = get_feature_details_from_path(
            "Base_2_2/Zone/ZoneBC/BottomLeft/PointList"
        )
        assert details["base"] == "Base_2_2"
        assert details["zone"] == "Zone"
        assert details["type"] == "tag"
        assert details["sub_type"] == "ZoneBC"
        assert details["name"] == "BottomLeft"

        with pytest.raises(AssertionError):
            get_feature_details_from_path("Dummy")

        with pytest.raises(ValueError):
            get_feature_details_from_path("Dummy/Dummy/Dummy/Dummy/Dummy/Dummy/Dummy")
