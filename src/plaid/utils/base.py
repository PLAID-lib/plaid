"""Base utilities."""
# %% Imports

from functools import wraps
import inspect
from typing import Type
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


def delegate_methods(to: str, delegate_cls: Type):
    """Class decorator to automatically forward all public methods from a delegate class."""

    # Programmatically extract all public methods from the class definition
    methods = [
        name for name, attr in inspect.getmembers(delegate_cls, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]

    def wrapper(cls):
        for name in methods:
            def make_delegate(method_name):
                target_method = getattr(delegate_cls, method_name)

                @wraps(target_method)
                def method(self, *args, **kwargs):
                    # Route execution to the instance attribute (e.g., self.features)
                    return getattr(getattr(self, to), method_name)(*args, **kwargs)
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
