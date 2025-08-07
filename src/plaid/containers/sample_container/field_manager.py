"""Module that implements the `FieldManager` class that holds the responsability of managing fields within a Sample."""

from plaid.types import CGNSTree, FieldType


class FieldManager:
    """Manager object for scalars."""

    def __init__(self):
        self.features: dict[str, FieldType] = {}

    def add_field(
        self,
        name: str,
        field: FieldType,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
    ) -> None:
        """Add field"."""
        pass

    def remove_field(
        self,
        name: str,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
    ) -> CGNSTree:
        """Remove field."""
        pass

    def get_field(
        self,
        name: str,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
    ) -> FieldType:
        """Get field."""
        pass

    def get_field_names(
        self,
        zone_name: str = None,
        base_name: str = None,
        location: str = "Vertex",
        time: float = None,
    ) -> set[str]:
        """Get all fields names."""
        pass
