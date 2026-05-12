"""Custom types for CGNS data structures."""
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
        """Validate CGNS path format.

        Args:
            v: Candidate CGNS path.

        Returns:
            The validated path.

        Raises:
            ValueError: If the path does not match the expected CGNS pattern.
        """
        if not CGNS_PATTERN.match(v):
            raise ValueError(
                "Invalid CGNS variable format. Need to be in the form of 'Base_X_Y/ZoneName/VariableName'"
            )
        return v

    @property
    def path(self) -> str:
        """Return the full CGNS path."""
        return self.root

    @property
    def base(self) -> str:
        """Return the base component of the CGNS path."""
        return self.root.split("/")[0]

    def zone(self) -> str:
        """Return the zone component of the CGNS path."""
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
