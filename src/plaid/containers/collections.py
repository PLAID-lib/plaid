"""Module for implementing collections of features within a Sample."""

import logging
from typing import Optional, Union

from plaid.types import Scalar

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)


def _check_names(names: Union[str, list[str]]):
    """Check that names do not contain invalid character ``/``.

    Args:
        names (Union[str, list[str]]): The names to check.

    Raises:
        ValueError: If any name contains the invalid character ``/``.
    """
    if isinstance(names, str):
        names = [names]
    for name in names:
        if (name is not None) and ("/" in name):
            raise ValueError(
                f"feature_names containing `/` are not allowed, but {name=}, you should first replace any occurence of `/` with something else, for example: `name.replace('/','__')`"
            )


class SampleScalars:
    """A container for scalar features within a Sample.

    Provides dict-like operations for adding, retrieving, and removing scalars.
    Names must be unique and may not contain the character ``/``.
    """

    def __init__(self, scalars: Optional[dict[str, Scalar]]) -> None:
        self.data: dict[str, Scalar] = scalars if scalars is not None else {}

    def add(self, name: str, value: Scalar) -> None:
        """Add a scalar value to a dictionary.

        Args:
            name (str): The name of the scalar value.
            value (Scalar): The scalar value to add or update in the dictionary.
        """
        _check_names(name)
        self.data[name] = value

    def remove(self, name: str) -> Scalar:
        """Delete a scalar value from the dictionary.

        Args:
            name (str): The name of the scalar value to be deleted.

        Raises:
            KeyError: Raised when there is no scalar / there is no scalar with the provided name.

        Returns:
            Scalar: The value of the deleted scalar.
        """
        if name not in self.data:
            raise KeyError(f"There is no scalar value with name {name}.")

        return self.data.pop(name)

    def get(self, name: str) -> Optional[Scalar]:
        """Retrieve a scalar value associated with the given name.

        Args:
            name (str): The name of the scalar value to retrieve.

        Returns:
            Scalar or None: The scalar value associated with the given name, or None if the name is not found.
        """
        return self.data.get(name)

    def get_names(self) -> list[str]:
        """Get a set of scalar names available in the object.

        Returns:
            list[str]: A set containing the names of the available scalars.
        """
        return sorted(self.data.keys())
