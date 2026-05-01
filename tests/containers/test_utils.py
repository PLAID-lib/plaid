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
    def test_get_feature_details_from_path(self, url, expected):
        assert get_feature_details_from_path(url) == expected


    def test_validate_required_infos(self):
        infos = {
            "legal": {"owner": "Joh Doe", "license": "cc-by-sa-4.0"},
        }
        validate_required_infos(infos)

        infos_missing_license = {
            "legal": {
                "owner": "Joh Doe",
            },
        }
        with pytest.raises(ValueError):
            validate_required_infos(infos_missing_license)

