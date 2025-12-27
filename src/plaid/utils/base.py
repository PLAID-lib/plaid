"""Base utilities."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from functools import wraps

import numpy as np

# %% Functions


def generate_random_ASCII(size: int = 16) -> str:
    """Generate a random ASCII string of the specified size.

    Args:
        size (int, optional): The length of the generated string. Defaults to 16.

    Returns:
        str: A random ASCII string of the specified size.
    """
    assert size >= 1
    rnd_ = chr(np.random.randint(65, 91))
    for _ in range(size - 1):
        val_ = np.random.randint(91 - 65 + 10)
        if val_ >= 10:
            rnd_ += chr(val_ - 10 + 65)
        else:
            rnd_ += str(val_)
    return rnd_


def safe_len(obj):
    """Safely return the length of an object, or 0 if the object has no length.

    Parameters
    ----------
    obj : Any
        The object whose length is to be computed.

    Returns:
    -------
    int
        The length of the object if it defines `__len__`, otherwise 0.
    """
    return len(obj) if hasattr(obj, "__len__") else 0


def delegate_methods(to: str, methods: list[str]):
    """Class decorator to forward specific methods from a delegate attribute."""

    def wrapper(cls):
        for name in methods:

            def make_delegate(name):
                @wraps(getattr(getattr(cls, to, None), name, lambda *_, **__: None))
                def method(self, *args, **kwargs):
                    return getattr(getattr(self, to), name)(*args, **kwargs)

                return method

            setattr(cls, name, make_delegate(name))
        return cls

    return wrapper


class NotAllowedError(Exception):
    """Exception for not allowed usage."""

    pass


class ShapeError(Exception):
    """Exception for badly shaped tensors."""

    pass


class DeprecatedError(Exception):
    """Exception for deprecated methods."""

    pass


def get_feature_type_by_path(path: str) -> str:
    """Retrieve a feature value type from a CGNS-style path.

    Args:
        path (str): CGNS node path relative to the mesh root (for example
            "BaseName/ZoneName/GridCoordinates/CoordinateX" or
            "BaseName/ZoneName/Solution/FieldName").
    """
    if path.startswith("Global/"):
        return "global"
    elif path.endswith("/ElementConnectivity"):
        return "element_connectivity"
    elif path.endswith("/ElementRange"):
        return "element_range"
    elif path.endswith("/PointList"):
        return "tag"
    elif path.endswith(("/CoordinateX", "/CoordinateY", "/CoordinateZ")):
        return "node_coordinate"
    else:
        return "field"


def get_feature_name_by_path(path: str) -> str:
    """Retrieve a feature name from a CGNS-style path.

    Args:
        path (str): CGNS node path relative to the mesh root (for example
            "BaseName/ZoneName/GridCoordinates/CoordinateX" or
            "BaseName/ZoneName/Solution/FieldName").
    """
    return path.split("/")[-1]
