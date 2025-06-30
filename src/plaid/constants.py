"""Common constants used across the PLAID library."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#


AUTHORIZED_TASKS = "regression", "classification"

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
"""List of element type names commonly used in Computational Fluid Dynamics (CFD). These names represent different types of finite elements that are used to discretize physical domains for numerical analysis."""

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
"""Configuration dictionary that specifies authorized information keys and their respective categories.
"""
