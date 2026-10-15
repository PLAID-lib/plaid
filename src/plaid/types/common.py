"""Common types used across the PLAID library."""

import sys
from typing import TYPE_CHECKING, Any, Union

if sys.version_info >= (3, 11):
    from typing import TypeAlias
else:  # pragma: no cover
    from typing_extensions import TypeAlias


import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# scalars
ScalarDType = Union[np.integer, np.floating, float]

# arrays
if TYPE_CHECKING:
    IArray = NDArray[np.integer[Any]]
    FArray = NDArray[np.floating[Any]]
    BytesS1Array = NDArray[np.bytes_]
else:
    # Keep runtime aliases simple so beartype does not interpret generic NumPy
    # scalar types as concrete dtypes.
    IArray = np.ndarray
    FArray = np.ndarray
    BytesS1Array = np.ndarray

Array: TypeAlias = IArray | FArray | np.integer | np.floating

# scalar or arrays
IScalarOrArray = int | np.integer | IArray
FScalarOrArray = float | np.floating | FArray
ScalarOrArray = ScalarDType | Array
ScalarOrArrayOrStr = ScalarDType | Array | str | BytesS1Array

# Types used in indexing operations
IndexArrayType = Union[list[int], IArray]
