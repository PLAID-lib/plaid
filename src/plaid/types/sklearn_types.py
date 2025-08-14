"""Custom types for scikit-learn related objects."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Union

from sklearn.base import (
    BaseEstimator,
    BiclusterMixin,
    ClassifierMixin,
    ClusterMixin,
    DensityMixin,
    MultiOutputMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
)

SklearnBlock = Union[
    BaseEstimator,
    TransformerMixin,
    RegressorMixin,
    ClassifierMixin,
    ClusterMixin,
    BiclusterMixin,
    DensityMixin,
    OutlierMixin,
    MultiOutputMixin,
]
