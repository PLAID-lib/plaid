"""Module that implements the `ScalarManager` class that holds the responsability of managing scalars within a Sample."""

from plaid.types import ScalarType


class ScalarManager:
    """Manager object for scalars."""

    def __init__(self):
        self.features: dict[str, ScalarType] = {}

    def add(self, name: str, value: ScalarType) -> None:
        """Add a scalar."""
        self.features[name] = value

    def remove(self, name: str) -> ScalarType:
        """Remove a scalar."""
        return self.features.pop(name)

    def get(self, name: str) -> ScalarType:
        """Get scalar."""
        return self.features[name]

    def get_names(self) -> set[str]:
        """Get all the scalars names."""
        return sorted(self.features.keys())
