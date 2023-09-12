# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from typing import TypedDict, Union

import numpy as np
from BasicTools.Containers.ConstantRectilinearMesh import \
    ConstantRectilinearMesh
from BasicTools.Containers.Filters import ElementFilter, NodeFilter
from BasicTools.Containers.UnstructuredMesh import UnstructuredMesh

# %% Functions


def generate_random_ASCII(size=16) -> str:
    """Generate a random ASCII string of the specified size.

    Args:
        size (int, optional): The length of the generated string. Defaults to 16.

    Returns:
        str: A random ASCII string of the specified size.
    """
    assert (size >= 1)
    rnd_ = chr(np.random.randint(65, 91))
    for _ in range(size - 1):
        val_ = np.random.randint(91 - 65 + 10)
        if val_ >= 10:
            rnd_ += chr(val_ - 10 + 65)
        else:
            rnd_ += str(val_)
    return rnd_

# %% Classes


BTMesh = Union[UnstructuredMesh, ConstantRectilinearMesh]
"""A BTMesh is an Union[UnstructuredMesh, ConstantRectilinearMesh]
"""
Filter = Union[ElementFilter, NodeFilter]
"""A Filter is an Union[ElementFilter, NodeFilter]
"""


class NotAllowedError(Exception):
    """Exception for not allowed usage."""
    pass


class ShapeError(Exception):
    """Exception for badly shaped tensors."""
    pass


class IdentifierError(Exception):
    """Exception for unavailable ``feature``/``storage`` with given ``identifier``."""
    pass

# TODO: use it instead of tuples (feature_type:str, feature_name:str)


class FeatureIndentifier(TypedDict):
    """Uniquely identifies ``feature``/``storage``."""
    type: str
    name: str

# TODO: use it instead of tuples (feature_type:str, feature_name:str,
# value_id:int)


class ValueIndentifier(TypedDict):
    """Uniquely identifies a ``value`` in a ``storage``."""
    type: str
    name: str
    id: int
