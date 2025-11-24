"""Wrapped scikit-learn transformers and regressors for PLAID Dataset compatibility.

Provides adapters to use scikit-learn estimators within the PLAID feature/block system:

- WrappedPlaidSklearnTransformer: wraps a TransformerMixin

- WrappedPlaidSklearnRegressor: wraps a RegressorMixin
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import copy
import sys
from typing import Optional

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")


from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.utils.validation import check_is_fitted

from plaid import Dataset
from plaid.containers import FeatureIdentifier
from plaid.containers.utils import check_features_type_homogeneity
from plaid.types import Array, SklearnBlock


def get_2Darray_from_homogeneous_identifiers(
    dataset: Dataset, features_identifiers: list[FeatureIdentifier]
) -> Array:
    """Returns a 2D array from a Dataset and a feature id.

    The function calls `dataset.get_tabular_from_homogeneous_identifiers(...)`, then removes
    either the second or third dimension if it has size 1, so that the output is 2D.

    Args:
        dataset (Dataset): A Dataset object exposing `get_tabular_from_homogeneous_identifiers`.
        features_identifiers (list[FeatureIdentifier]): a list of input feature identifiers.

    Returns:
        A NumPy array of shape (n_samples, n_features).

    Raises:
        AssertionError: If the number of features in the output does not match the identifiers.
        ValueError: If both the second and third dimensions have size greater than 1.
    """
    X = dataset.get_tabular_from_homogeneous_identifiers(features_identifiers)
    # X is of size (nb_sample, nb_features, dim_features), either nb_features or dim_features should be 1 to be compatible with scikit-learn blocks
    if X.shape[1] == 1:
        X = X[:, 0, :]
    elif X.shape[2] == 1:
        X = X[:, :, 0]
    else:
        raise ValueError(
            "X (generate by dataset.get_tabular_from_homogeneous_identifiers) is expected to have its second or third dimension equal to 1"
        )

    return X


class WrappedSklearnTransformer(TransformerMixin, BaseEstimator):
    """Adapter for using a scikit-learn transformer on PLAID Datasets.

    Transforms tabular data extracted from homogeneous feature identifiers,
    and returns results as a `Dataset`. Supports forward and inverse transforms.

    Args:
        sklearn_block (SklearnBlock): A scikit-learn Transformer implementing fit/transform APIs.
        in_features_identifiers (list[FeatureIdentifier]): List of feature identifiers to extract input data from.
        out_features_identifiers (list[FeatureIdentifier], optional): List of feature identifiers used for outputs. If None,
            defaults to `in_features_identifiers`.
    """

    # TODO: check if restrict_to_features=True can be used to reduce further memory consumption
    def __init__(
        self,
        sklearn_block: SklearnBlock,
        in_features_identifiers: list[FeatureIdentifier],
        out_features_identifiers: Optional[list[FeatureIdentifier]] = None,
    ):
        self.sklearn_block = sklearn_block
        self.in_features_identifiers = in_features_identifiers
        self.out_features_identifiers = out_features_identifiers

    def fit(self, dataset: Dataset, _y=None) -> Self:
        """Fits the underlying scikit-learn transformer on selected input features.

        Args:
            dataset: A `Dataset` object containing the features to transform.
            _y: Ignored.

        Returns:
            self: The fitted transformer.
        """
        self.in_features_identifiers_ = copy.deepcopy(self.in_features_identifiers)
        check_features_type_homogeneity(self.in_features_identifiers_)

        if self.out_features_identifiers:
            self.out_features_identifiers_ = copy.deepcopy(
                self.out_features_identifiers
            )
            check_features_type_homogeneity(self.out_features_identifiers_)
        else:
            self.out_features_identifiers_ = copy.deepcopy(self.in_features_identifiers)

        X = get_2Darray_from_homogeneous_identifiers(
            dataset, self.in_features_identifiers_
        )

        self.sklearn_block_ = clone(self.sklearn_block).fit(X, _y)

        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """Applies the fitted transformer to the selected input features.

        Args:
            dataset: A `Dataset` object to transform.

        Returns:
            Dataset: Transformed features wrapped as a new `Dataset`.
        """
        check_is_fitted(self, "sklearn_block_")

        X = get_2Darray_from_homogeneous_identifiers(
            dataset, self.in_features_identifiers_
        )

        X_transformed = self.sklearn_block_.transform(X)
        X_transformed = X_transformed.reshape(
            (len(dataset), len(self.out_features_identifiers_), -1)
        )

        dataset_transformed = dataset.add_features_from_tabular(
            X_transformed, self.out_features_identifiers_, restrict_to_features=False
        )

        return dataset_transformed

    def inverse_transform(self, dataset: Dataset) -> Dataset:
        """Applies inverse transformation to the output features.

        Args:
            dataset: A `Dataset` object with transformed output features.

        Returns:
            Dataset: Dataset with inverse-transformed features.
        """
        check_is_fitted(self, "sklearn_block_")

        X = get_2Darray_from_homogeneous_identifiers(
            dataset, self.out_features_identifiers_
        )

        X_inv_transformed = self.sklearn_block_.inverse_transform(X)

        X_inv_transformed = X_inv_transformed.reshape(
            (len(dataset), len(self.in_features_identifiers_), -1)
        )

        dataset_inv_transformed = dataset.add_features_from_tabular(
            X_inv_transformed, self.in_features_identifiers_, restrict_to_features=False
        )

        return dataset_inv_transformed


class WrappedSklearnRegressor(RegressorMixin, BaseEstimator):
    """Adapter for using a scikit-learn regressor with PLAID Dataset.

    Fits and predicts on tabular arrays extracted from stacked features,
    while preserving the feature/block structure expected by PLAID.

    Args:
        sklearn_block: A scikit-learn regressor with fit/predict API.
        in_features_identifiers: List of feature identifiers for inputs.
        out_features_identifiers: List of feature identifiers for outputs.
    """

    # TODO: remove transform and inv tranf

    def __init__(
        self,
        sklearn_block: SklearnBlock,
        in_features_identifiers: list[FeatureIdentifier],
        out_features_identifiers: list[FeatureIdentifier],
    ):
        self.sklearn_block = sklearn_block
        self.in_features_identifiers = in_features_identifiers
        self.out_features_identifiers = out_features_identifiers

    def fit(self, dataset: Dataset, _y=None) -> Self:
        """Fits the wrapped scikit-learn regressor on the stacked input/output data.

        Args:
            dataset: A `Dataset` containing both input and output features.
            _y: Ignored.

        Returns:
            self: The fitted regressor.
        """
        self.sklearn_block_ = clone(self.sklearn_block)
        self.in_features_identifiers_ = self.in_features_identifiers.copy()
        self.out_features_identifiers_ = self.out_features_identifiers.copy()

        X, _ = dataset.get_tabular_from_stacked_identifiers(
            self.in_features_identifiers_
        )
        y, self.cumulated_feat_dims = dataset.get_tabular_from_stacked_identifiers(
            self.out_features_identifiers_
        )

        self.sklearn_block_.fit(X, y)

        return self

    def predict(self, dataset: Dataset) -> Dataset:
        """Predicts target values using the fitted regressor.

        Args:
            dataset: A `Dataset` with input features.

        Returns:
            Dataset: A new `Dataset` containing predicted target features.
        """
        check_is_fitted(self, "sklearn_block_")

        X, _ = dataset.get_tabular_from_stacked_identifiers(
            self.in_features_identifiers_
        )

        y = self.sklearn_block_.predict(X)
        y = y.reshape((len(dataset), -1))

        dataset_predicted = Dataset.merge_dataset_by_features(
            [
                dataset.from_tabular(
                    y[
                        :,
                        None,
                        self.cumulated_feat_dims[i_feat] : self.cumulated_feat_dims[
                            i_feat + 1
                        ],
                    ],
                    feature_identifiers=[feat_ids],
                )
                for i_feat, feat_ids in enumerate(self.out_features_identifiers_)
            ]
        )
        # dataset_predicted = dataset.add_features_from_tabular(
        #     y, self.out_features_identifiers_, restrict_to_features=False
        # )
        dataset_predicted = dataset.merge_features(dataset_predicted)

        return dataset_predicted
