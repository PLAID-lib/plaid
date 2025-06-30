"""Common types used across the PLAID library."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

# A generic float array type (float32 or float64)
ArrayDType = Union[np.float32, np.float64]
Array: TypeAlias = NDArray[ArrayDType]

# Types used in indexing operations
IndexType = Union[list[int], Array]
