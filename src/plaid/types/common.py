"""Common types used across the PLAID library."""

from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# scalars
ScalarDType = Union[np.integer, np.floating, int, float]

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

Array = IArray | FArray | np.integer | np.floating

# scalar or arrays
IScalarOrArray = int | np.integer | IArray
FScalarOrArray = float | np.floating | FArray
ScalarOrArray = ScalarDType | Array
# np.bytes_ / np.bool_: size-1 str or bool globals, unwrapped by Sample.get_global
ScalarOrArrayOrStr = ScalarDType | Array | str | BytesS1Array | np.bytes_ | np.bool_

# Types used in indexing operations
IndexArrayType = Union[list[int], IArray]
