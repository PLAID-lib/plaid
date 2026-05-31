"""Custom types for CGNS data structures."""

import sys
from typing import Any, List

if sys.version_info >= (3, 11):
    from typing import TypeAlias
else:  # pragma: no cover
    from typing_extensions import TypeAlias


# A CGNSTree is simply the root CGNSNode
# CGNSTree: TypeAlias = CGNSNode

# CGNSTree: TypeAlias = tuple[str, Union[np.ndarray,float, bool, None], List["CGNSTree"], str]
# normally we need a tuple but pycgns uses a list
# CGNSTree: TypeAlias = List[str | np.ndarray | np.floating | bool| None | List["plaid.types.cgns_types.CGNSTree"] | str]
# beartype gets really confusing so for the moment me use a very generic type
CGNSTree: TypeAlias = List[Any]
CGNSNode: TypeAlias = CGNSTree
