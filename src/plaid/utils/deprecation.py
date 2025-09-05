"""Implementation of deprecation utilities."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import functools
import warnings
from typing import Optional

try:
    from warnings import deprecated as deprecated_builtin  # Python 3.13+
except ImportError:
    deprecated_builtin = None

# %% Functions


def deprecated(
    reason: str, version: Optional[str] = None, removal: Optional[str] = None
):
    """Decorator to mark a function, method, or class as deprecated.

    Uses built-in `warnings.deprecated` when running on Python 3.13+,
    otherwise falls back to a custom warning wrapper.

    Args:
        reason (str): Explanation and suggested replacement.
        version (str, optional): Version since deprecation.
        removal (str, optional): Planned removal version.
    """
    message_parts = [reason]
    if version:
        message_parts.append(f"[since v{version}]")
    if removal:
        message_parts.append(f"(will be removed in v{removal})")
    full_message = " ".join(message_parts)

    if deprecated_builtin is not None:  # pragma: no cover

        def decorator(obj):
            return deprecated_builtin(
                full_message, category=DeprecationWarning, stacklevel=2
            )(obj)

        return decorator

    def decorator(obj):
        if isinstance(obj, type):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(full_message, DeprecationWarning, stacklevel=2)
                return orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj

        elif callable(obj):

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(full_message, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return wrapper

        else:
            raise TypeError(
                "@deprecated decorator with non-None category must be applied to "
                f"a class or callable, not {obj!r}"
            )

    return decorator


def deprecated_argument(
    old_arg: str,
    new_arg: str,
    converter=lambda x: x,
    version: Optional[str] = None,
    removal: Optional[str] = None,
):
    """Decorator to mark a function argument as deprecated and redirect it to a new argument.

    Args:
        old_arg (str): Name of the old argument.
        new_arg (str): Name of the new argument.
        converter (callable): Function to convert the old value into the new format.
        version (str, optional): Version since deprecation.
        removal (str, optional): Planned removal version.
    """
    message_parts = [f"Argument `{old_arg}` is deprecated, use `{new_arg}` instead."]
    if version:
        message_parts.append(f"[since v{version}]")
    if removal:
        message_parts.append(f"(will be removed in v{removal})")
    full_message = " ".join(message_parts)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old_arg in kwargs:
                # Emit deprecation warning
                if deprecated_builtin is not None:  # pragma: no cover
                    # In Python 3.13+, link warning to the function itself
                    decorated = deprecated_builtin(
                        full_message, category=DeprecationWarning, stacklevel=2
                    )(func)
                    return decorated(
                        *args, **{new_arg: converter(kwargs.pop(old_arg)), **kwargs}
                    )
                else:
                    warnings.warn(full_message, DeprecationWarning, stacklevel=2)
                kwargs[new_arg] = converter(kwargs.pop(old_arg))
            return func(*args, **kwargs)

        return wrapper

    return decorator
