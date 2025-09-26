"""Custom types for CGNS data structures."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import sys
from typing import Any, Optional

from pydantic import BaseModel, Field

if sys.version_info >= (3, 11):
    from typing import TypeAlias
else:  # pragma: no cover
    from typing_extensions import TypeAlias


class CGNSNode(BaseModel):
    """Custom type for a CGNS node."""

    name: str = Field(..., description="The name of the CGNS node.")
    value: Optional[Any] = Field(
        None,
        description="The value of the CGNS node, which can be of any type or None.",
    )
    children: list["CGNSNode"] = Field(
        default_factory=list, description="A list of child CGNS nodes."
    )
    label: str = Field(..., description="The label of the CGNS node.")


# A CGNSTree is simply the root CGNSNode
CGNSTree: TypeAlias = CGNSNode
