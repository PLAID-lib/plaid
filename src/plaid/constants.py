"""Common constants used across the PLAID library.

AUTHORIZED_TASKS: Tuple of strings representing the types of tasks supported by PLAID.

CGNS_ELEMENT_NAMES: List of strings representing the names of CGNS elements.

AUTHORIZED_INFO_KEYS: Dictionary defining the keys allowed in different sections of metadata.
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

AUTHORIZED_TASKS = ["regression", "classification"]

AUTHORIZED_FEATURE_TYPES = ["scalar", "time_series", "field", "nodes"]

AUTHORIZED_FEATURE_INFOS = {
    "scalar": ["name"],
    "time_series": ["name"],
    "field": ["name", "base_name", "zone_name", "location", "time"],
    "nodes": ["base_name", "zone_name", "time"],
}

AUTHORIZED_FIELD_LOCATIONS = ["Vertex", "EdgeCenter", "FaceCenter", "CellCenter"]

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
