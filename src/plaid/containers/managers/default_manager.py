"""DefaultManager for managing default time/base/zone selections."""

from dataclasses import dataclass
from typing import Optional, Protocol


class FeaturesBackend(Protocol):
    """Minimal interface required by DefaultManager.

    This allows any backend implementing these methods to be used by DefaultManager.

    For example, SampleFeatures already satisfies this contract.
    """

    def get_all_time_values(self) -> list[float]:
        """Get all available time values in the mesh."""
        ...

    def get_base_names(self, *, time: Optional[float] = None) -> list[str]:
        """Get all available base names at a given time."""
        ...

    def get_zone_names(
        self, base_name: Optional[str] = None, *, time: Optional[float] = None
    ) -> list[str]:
        """Get all available zone names within a base at a given time."""
        ...

    def has_base(self, base_name: str, time: Optional[float] = None) -> bool:
        """Check if a base exists at a given time."""
        ...

    def has_zone(
        self,
        zone_name: str,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
    ) -> bool:
        """Check if a zone exists within a base at a given time."""
        ...


@dataclass
class DefaultManager:
    """Manages and resolves default time/base/zone selections.

    Notes on legacy semantics:
    - resolve_time(None) returns the first sorted available timestamp, else 0.0.
    - resolve_base(None) returns None if no non-Global base exists.
    - resolve_zone(None) returns None if no zone exists in the resolved base.
    """

    _features: FeaturesBackend

    _default_active_base: Optional[str] = None
    _default_active_zone: Optional[str] = None
    _default_active_time: Optional[float] = None

    def set_default_time(self, time: float) -> None:
        """Set the default time for the system.

        This function sets the default time to be used for various operations in the system.

        Args:
            time (float): The time value to be set as the default.

        Raises:
            ValueError: If the specified time does not exist in the available mesh times.

        Note:
            - Setting the default time is important for synchronizing operations with a specific time point in the system's data.
            - The available mesh times can be obtained using the `get_all_mesh_times` method.

        Example:
            .. code-block:: python

                from plaid import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)
                print(sample.show_tree(0.5))
                >>> ...

                # Set the default time to 0.5 seconds
                sample.set_default_time(0.5)

                # You can now use class functions with 0.5 as default time
                print(sample.show_tree()) # show the cgns tree at the time 0.5
                >>> ...
        """
        if time in (self._default_active_time, None):
            return
        if time not in self._features.get_all_time_values():
            raise ValueError(f"time {time} does not exist in mesh times")
        self._default_active_time = time

    def set_default_base(self, base_name: str, time: Optional[float] = None) -> None:
        """Set the default base for the specified time (that will also be set as default if provided).

        The default base is a reference point for various operations in the system.

        Args:
            base_name (str): The name of the base to be set as the default.
            time (float, optional): The time at which the base should be set as default. If not provided, the default base and active zone will be set with the default time.

        Raises:
            ValueError: If the specified base does not exist at the given time.

        Note:
            - Setting the default base and is important for synchronizing operations with a specific base in the system's data.
            - The available mesh base can be obtained using the `get_base_names` method.

        Example:
            .. code-block:: python

                from plaid import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)
                print(sample.get_physical_dim("BaseA", 0.5))
                >>> 3

                # Set "BaseA" as the default base for the default time
                sample.set_default_base("BaseA")

                # You can now use class functions with "BaseA" as default base
                print(sample.get_physical_dim(0.5))
                >>> 3

                # Set "BaseB" as the default base for a specific time
                sample.set_default_base("BaseB", 0.5)

                # You can now use class functions with "BaseB" as default base and 0.5 as default time
                print(sample.get_physical_dim()) # Physical dim of the base "BaseB"
                >>> 3
        """
        if time is not None:
            self.set_default_time(time)
        if base_name in (self._default_active_base, None):
            return
        if not self._features.has_base(base_name, time):
            raise ValueError(f"base {base_name} does not exist at time {time}")
        self._default_active_base = base_name

    def set_default_zone_base(
        self, zone_name: str, base_name: str, time: Optional[float] = None
    ) -> None:
        """Set the default base and active zone for the specified time (that will also be set as default if provided).

        The default base and active zone serve as reference points for various operations in the system.

        Args:
            zone_name (str): The name of the zone to be set as the active zone.
            base_name (str): The name of the base to be set as the default.
            time (float, optional): The time at which the base and zone should be set as default. If not provided, the default base and active zone will be set with the default time.

        Raises:
            ValueError: If the specified base or zone does not exist at the given time

        Note:
            - Setting the default base and zone are important for synchronizing operations with a specific base/zone in the system's data.
            - The available mesh bases and zones can be obtained using the `get_base_names` and `get_base_zones` methods, respectively.

        Example:
            .. code-block:: python

                from plaid import Sample
                sample = Sample("path_to_plaid_sample")
                print(sample)
                >>> Sample(2 scalars, 1 timestamp, 5 fields)
                print(sample.get_zone_type("ZoneX", "BaseA", 0.5))
                >>> Structured

                # Set "BaseA" as the default base and "ZoneX" as the active zone for the default time
                sample.set_default_zone_base("ZoneX", "BaseA")

                # You can now use class functions with "BaseA" as default base with "ZoneX" as default zone
                print(sample.get_zone_type(0.5)) # type of the zone "ZoneX" of base "BaseA"
                >>> Structured

                # Set "BaseB" as the default base and "ZoneY" as the active zone for a specific time
                sample.set_default_zone_base("ZoneY", "BaseB", 0.5)

                # You can now use class functions with "BaseB" as default base with "ZoneY" as default zone and 0.5 as default time
                print(sample.get_zone_type()) # type of the zone "ZoneY" of base "BaseB" at 0.5
                >>> Unstructured
        """
        self.set_default_base(base_name, time)
        if zone_name in (self._default_active_zone, None):
            return
        if not self._features.has_zone(zone_name, base_name, time):
            raise ValueError(
                f"zone {zone_name} does not exist for the base {base_name} at time {time}"
            )
        self._default_active_zone = zone_name

    def resolve_time(self, time: Optional[float] = None) -> float:
        """Resolve which time to use for an operation.

        Resolution order:
        - If `time` is provided, return it.
        - Else if a default time is set, return it.
        - Else return the first sorted available timestamp, or 0.0 if none exist.

        Args:
            time (float, optional): The time to resolve. Defaults to None.

        Returns:
            float: The resolved time.
        """
        if time is not None:
            return time

        if self._default_active_time is not None:
            return self._default_active_time

        timestamps = self._features.get_all_time_values()
        return sorted(timestamps)[0] if timestamps else 0.0

    def resolve_base(
        self, base_name: Optional[str] = None, time: Optional[float] = None
    ) -> Optional[str]:
        """Resolve which base name to use for an operation.

        Args:
            base_name (str, optional): The base name to resolve. Defaults to None.
            time (float, optional): The time at which to resolve the base. Defaults to None.

        Returns:
            Optional[str]: The resolved base name.

        Raises:
            KeyError: If multiple bases exist and no default is set.
        """
        base_name = base_name or self._default_active_base
        if base_name:
            return base_name

        base_names = self._features.get_base_names(time=time)
        if "Global" in base_names:
            base_names.remove("Global")

        if len(base_names) == 0:
            return None
        if len(base_names) == 1:
            return base_names[0]

        raise KeyError(f"No default base provided among {base_names}")

    def resolve_zone(
        self,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
    ) -> Optional[str]:
        """Resolve which zone name to use for an operation.

        Args:
            zone_name (str, optional): The zone name to resolve. Defaults to None.
            base_name (str, optional): The base name in which the zone is located. Defaults to None.
            time (float, optional): The time at which to resolve the zone. Defaults

        Returns::
            Optional[str]: The resolved zone name.

        Raises:
            KeyError: If multiple zones exist and no default is set.
        """
        zone_name = zone_name or self._default_active_zone
        if zone_name:
            return zone_name

        resolved_base = self.resolve_base(base_name, time)
        zone_names = self._features.get_zone_names(resolved_base, time=time)

        if len(zone_names) == 0:
            return None
        if len(zone_names) == 1:
            return zone_names[0]

        raise KeyError(
            f"No default zone provided among {zone_names} in the default base: {resolved_base}"
        )
