"""Deprecation utilities for PLAID."""

import functools
import warnings


def deprecated_feature_names(alternative: str):
    """Decorator to mark feature name-based methods as deprecated.

    Args:
        alternative (str): The name of the alternative feature identifier-based method to use.

    Returns:
        callable: The decorated function that will emit a deprecation warning.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in a future version. "
                f"Use {alternative} with feature identifiers instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator
