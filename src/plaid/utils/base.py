# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

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


class NotAllowedError(Exception):
    """Exception for not allowed usage."""

    pass


class ShapeError(Exception):
    """Exception for badly shaped tensors."""

    pass


class DeprecatedError(Exception):
    """Exception for deprecated methods."""

    pass
