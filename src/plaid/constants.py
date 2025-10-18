# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
"""This module defines common constants used throughout the PLAID library.

It includes:
- AUTHORIZED_TASKS: List of supported task types (e.g., regression, classification).
- AUTHORIZED_FEATURE_TYPES: List of supported feature types (e.g., scalar, field, nodes).
- AUTHORIZED_FEATURE_INFOS: Dictionary specifying allowed metadata keys for various feature types.
- AUTHORIZED_INFO_KEYS: Dictionary specifying allowed metadata keys for various information sections.
- CGNS_FIELD_LOCATIONS: List of valid field locations as defined by the CGNS standard.
- CGNS_ELEMENT_NAMES: List of CGNS element names representing different mesh element types.

These constants help standardize metadata, task types, and mesh element references across the PLAID codebase.
"""

AUTHORIZED_TASKS = ["regression", "classification"]

AUTHORIZED_SCORE_FUNCTIONS = ["RRMSE"]

AUTHORIZED_FEATURE_TYPES = ["scalar", "field", "nodes"]

AUTHORIZED_FEATURE_INFOS = {
    "scalar": ["name"],
    "field": ["name", "location", "zone_name", "base_name", "time"],
    "nodes": ["zone_name", "base_name", "time"],
}

# Information keys for dataset metadata
# key ["plaid"]["version"] is not included as it is managed internally
AUTHORIZED_INFO_KEYS = {
    "legal": ["owner", "license"],
    "data_production": [
        "owner",
        "license",
        "type",
        "physics",
        "simulator",
        "hardware",
        "computation_duration",
        "script",
        "contact",
        "location",
    ],
    "data_description": [
        "number_of_samples",
        "number_of_splits",
        "DOE",
        "inputs",
        "outputs",
    ],
}

# See https://cgns.org/standard/SIDS/grid.html#flow-solution-structure-definition-flowsolution-t
CGNS_FIELD_LOCATIONS = [
    "Vertex",
    "CellCenter",
    "FaceCenter",
    "IFaceCenter",
    "JFaceCenter",
    "KFaceCenter",
    "EdgeCenter",
]

CGNS_ELEMENT_NAMES = [
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
    "HEXA_64",
]
