"""Common types used across the PLAID library."""

import sys
from typing import Union

if sys.version_info >= (3, 11):
    from typing import TypeAlias
else:  # pragma: no cover
    from typing_extensions import TypeAlias


import numpy as np
from numpy.typing import NDArray

# scalars
ScalarDType = Union[np.integer, np.floating, float]

# arrays
IArray = NDArray[np.integer]
FArray = NDArray[np.floating]
Array: TypeAlias = IArray | FArray | np.integer | np.floating
BytesS1Array = NDArray[np.dtype("S1")]

# scalar or arrays
IScalarOrArray = int | np.integer | IArray
FScalarOrArray = float | np.floating | FArray
ScalarOrArray = ScalarDType | Array
ScalarOrArrayOrStr = ScalarDType | Array | str | BytesS1Array

# Types used in indexing operations
IndexArrayType = Union[list[int], IArray]
