"""Custom types for CGNS data structures."""

from typing import Any

# A CGNSTree is simply the root CGNSNode
# CGNSTree: TypeAlias = CGNSNode

# CGNSTree: TypeAlias = tuple[str, Union[np.ndarray,float, bool, None], List["CGNSTree"], str]
# normally we need a tuple but pycgns uses a list
# CGNSTree: TypeAlias = List[str | np.ndarray | np.floating | bool| None | List["plaid.types.cgns_types.CGNSTree"] | str]
# beartype gets really confusing so for the moment me use a very generic type
CGNSTree = list[Any]
CGNSNode = CGNSTree
