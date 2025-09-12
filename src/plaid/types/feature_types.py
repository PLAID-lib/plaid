"""Custom types for features."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Union

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

from typing import Annotated, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic import Field as PydField

from plaid.types.common import Array


# Feature identifiers
class Identifier(BaseModel):
    """Feature identifier definas as a Pydantic model.

    Attributes:
        type: The type of feature (scalar, time_series, field, nods)
        name: The name of the feature (e.g., pressure)

    """

    type: Literal["scalar", "time_series", "field", "nodes"]
    name: str


class ScalarIdentifier(Identifier):
    """Feature identifier for scalars."""

    type = "scalar"


class TimeSeriesIdentifier(Identifier):
    """Feature identifier for time series."""

    type = "time_series"


class FieldIdentifier(Identifier):
    """Feature identifier for fields.

    Attributes:
        base_name: Optional base name (default: 'Base')
        zone_name: Optional zone name (default: 'Zone')
        location: Optional location (default: 'Vertex')
        time: Optional time value (default: 0.0). Must be greater than one.
    """

    type = "field"
    base_name: Optional[str] = PydField(default="Base")
    zone_name: Optional[str] = PydField("Zone")
    location: Optional[str] = PydField("Vertex")
    time: Optional[float] = PydField(default=0.0, ge=0.0)


class NodesIdentifier(Identifier):
    """Feature identifier for nodes.

    Attributes:
        base_name: Optional[str] = PydField(default="Base")
        zone_name: Optional[str] = PydField("Zone")
        time: Optional[float] = PydField(default=0.0, ge=0.0)
    """

    type = "nodes"
    base_name: Optional[str] = PydField(default="Base")
    zone_name: Optional[str] = PydField(default="Zone")
    time: Optional[float] = PydField(default=0.0, ge=0.0)


# Discreminate feature identifiers by their types
FeatureIdentifier = Annotated[
    Union[ScalarIdentifier, TimeSeriesIdentifier, FieldIdentifier, NodesIdentifier],
    PydField(discriminator="type"),
]


# Feature types
class Scalar(BaseModel):
    """Scalar feature.

    Attributes:
        identifier: A `ScalarIdentifier`
        value: Value of the scalar (int or float)
    """

    identifier: ScalarIdentifier
    value: Union[int, float]

    @property
    def name(self):
        """Returns the name of the scalar."""
        return self.identifier.name


class Field(BaseModel):
    """Field feature.

    Attributes:
        identifier: A `FieldIdentifier`
        value: The values of the field given as an array of shape (N, 1)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    identifier: FieldIdentifier
    value: NDArray[np.float64]

    @field_validator("value")
    def _check_value_shape(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if v.ndim != 2 or v.shape[1] != 1:
            raise ValueError(f"value must have shape [N, 1], got {v.shape}")
        return v

    @property
    def name(self):
        """Returns the name of the field."""
        return self.identifier.name


class Nodes(BaseModel):
    """Field feature.

    Attributes:
        identifier: A `FieldIdentifier`
        value: The coordinates of the nodes given as an array of shape (N, d) with 1 <= d <= 3.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    identifier: NodesIdentifier
    value: NDArray[np.float64]

    @field_validator("value")
    def _check_value_shape(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if v.ndim != 2 or v.shape[1] <= 3:
            raise ValueError(f"value must have shape [N, 1], got {v.shape}")
        return v

# TimeSeries might be removed soon: using aliases.
TimeSequence: TypeAlias = Array
TimeSeries: TypeAlias = tuple[TimeSequence, Field]

# Union of possible features.
Feature = Union[Scalar, Field, TimeSeries, Nodes]
