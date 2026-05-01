"""Custom types for CGNS data structures."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import sys
from typing import Any, Optional

from pydantic import BaseModel, Field, RootModel, field_validator

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

import re

CGNS_PATTERN = re.compile(r"^Base_\d+_\d+/[^/]+/[^/]+$")

class CGNSPath(RootModel):
    root: str

    @field_validator("root")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not CGNS_PATTERN.match(v):
            raise ValueError("Invalid CGNS variable format. Need to be in the form of 'Base_X_Y/ZoneName/VariableName'")
        return v

    @property
    def path(self) -> str:
        return self.root
    
    @property
    def base(self) -> str:
        return self.root.split("/")[0]
    
    def zone(self) -> str:
        return self.root.split("/")[1]
    

# Example usage of CGNSPath
if __name__ == "__main__":
    # Valid CGNS paths
    valid_path = CGNSPath("Base_1_0/Zone/GridCoordinates")
    print(f"Valid path: {valid_path.root}")

    valid_path2 = CGNSPath("Base_0_0/Normal/Normals")
    print(f"Valid path: {valid_path2.root}")
    print(f"Valid path: {valid_path2.path}")

    # Invalid CGNS paths will raise ValidationError
    try:
        invalid_path = CGNSPath("InvalidPath")
    except Exception as e:
        print(f"Invalid path error: {e}")

