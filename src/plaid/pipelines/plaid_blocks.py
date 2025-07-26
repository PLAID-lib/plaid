"""Custom meta-estimators for applying feature-wise and target-wise transformations.

Includes:
- PlaidTransformedTargetRegressor: transforms the target before fitting.
- PlaidColumnTransformer: applies transformers to feature subsets like ColumnTransformer.
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import copy

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted

from plaid.containers.dataset import Dataset
from plaid.containers.utils import has_duplicates_feature_ids


class PlaidColumnTransformer(ColumnTransformer):
    """Custom column-wise transformer for PLAID-style datasets.

    Similar to scikit-learn's `ColumnTransformer`, this class applies a list
    of transformer blocks to subsets of features, defined by their feature
    identifiers. Additionally, it preserves a set of remainder features that
    bypass transformation.

    Args:
        plaid_transformers: A list of tuples
            (name, transformer), where each `transformer` is a TransformerMixin.

    Notes:
        At fit, it is checked that `plaid_transformers` share no in_features_identifiers and no out_features_identifiers.
    """

    def __init__(self, plaid_transformers: list[tuple[str, TransformerMixin]] = None):
        self.plaid_transformers = plaid_transformers

        if plaid_transformers:
            transformers_dummy = [
                (name, transformer, "_") for name, transformer in plaid_transformers
            ]
        else:
            transformers_dummy = None

        super().__init__(transformers_dummy)

    def fit(self, dataset, _y=None):
        """Fits all transformers on their corresponding feature subsets.

        Args:
            dataset: A `Dataset` object or a list of samples.
            y: Ignored. Present for API compatibility.

        Returns:
            self: The fitted PlaidColumnTransformer.
        """
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

        self.in_features_identifiers_ = []
        for _, transformer in self.plaid_transformers:
            self.in_features_identifiers_ += copy.deepcopy(
                transformer.in_features_identifiers
            )

        assert not has_duplicates_feature_ids(self.in_features_identifiers_), (
            "Identical in_features_identifiers found among provided transformer: not compatible with PlaidColumnTransformer."
        )

        self.plaid_transformers_ = [
            (copy.deepcopy(name), clone(transformer))
            for name, transformer in self.plaid_transformers
        ]

        self.transformers_ = []
        for name, transformer in self.plaid_transformers_:
            sub_dataset = dataset.from_features_identifier(
                transformer.in_features_identifiers
            )
            transformer_ = clone(transformer).fit(sub_dataset)
            self.transformers_.append((name, transformer_, "_"))

        self.out_features_identifiers_ = []
        for _, transformer, _ in self.transformers_:
            self.out_features_identifiers_ += copy.deepcopy(
                transformer.out_features_identifiers_
            )

        assert not has_duplicates_feature_ids(self.out_features_identifiers_), (
            "Identical out_features_identifiers found among provided transformer: not compatible with PlaidColumnTransformer."
        )

        return self

    def transform(self, dataset):
        """Applies fitted transformers to feature subsets and merges results.

        Args:
            dataset: A `Dataset` object or a list of samples.

        Returns:
            Dataset: A new `Dataset` with transformed feature blocks, including
            untransformed remainder features.
        """
        check_is_fitted(self, "transformers_")
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

        transformed_datasets = [dataset.copy()]
        for _, transformer_, _ in self.transformers_:
            sub_dataset = dataset.from_features_identifier(
                transformer_.in_features_identifiers
            )
            transformed = transformer_.transform(sub_dataset)
            transformed_datasets.append(transformed)
        return Dataset.merge_dataset_by_features(transformed_datasets)

    def fit_transform(self, X, y=None):
        """Fits all transformers and returns the combined transformed dataset.

        Args:
            X: A `Dataset` object or a list of samples.
            y: Ignored. Present for API compatibility.

        Returns:
            Dataset: A new `Dataset` with transformed features.
        """
        return self.fit(X, y).transform(X)


class PlaidTransformedTargetRegressor(RegressorMixin, BaseEstimator):
    """Meta-estimator that transforms the target before fit and inverses it at predict.

    This regressor is compatible with custom `Dataset` objects and supports
    complex targets, including scalars and fields. It wraps a base regressor
    and a transformer that is responsible for preprocessing the target space.

    Args:
        regressor: A regressor implementing `fit` and `predict`, following the scikit-learn API.
        transformer: A transformer implementing `fit`, `transform`, and `inverse_transform`.
            Applied to the dataset before fitting the regressor.
    """

    def __init__(
        self,
        regressor: RegressorMixin = None,
        transformer: TransformerMixin = None,
    ):
        self.regressor = regressor
        self.transformer = transformer

    def fit(self, dataset, _y=None):
        """Fits the transformer and the regressor on the transformed dataset.

        Args:
            dataset: A `Dataset` object or a list of sample dictionaries.
                Input training data.
            y: Ignored. Present for API compatibility.

        Returns:
            self: The fitted estimator.
        """
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

        self.transformer_ = clone(self.transformer).fit(dataset)

        transformed_dataset = self.transformer_.transform(dataset)
        self.regressor_ = clone(self.regressor).fit(transformed_dataset)

        self.in_features_identifiers_ = copy.deepcopy(
            self.regressor_.in_features_identifiers_
        )
        self.out_features_identifiers_ = copy.deepcopy(
            self.transformer_.in_features_identifiers_
        )

        return self

    def predict(self, dataset):
        """Predicts target values using the fitted regressor, then applies the inverse transformation.

        Args:
            dataset: A `Dataset` object or a list of sample dictionaries.
                Input data to predict on.

        Returns:
            Dataset: A `Dataset` containing the inverse-transformed predictions.
        """
        check_is_fitted(self, "regressor_")
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        dataset_pred_transformed = self.regressor_.predict(dataset)
        return self.transformer_.inverse_transform(dataset_pred_transformed)

    def score(self, dataset_X, dataset_y=None):
        """Computes a normalized root mean squared error (RMSE) score on the transformed targets.

        The score is defined as `1 - avg(relative RMSE)` over all target features in the
        `transformer` input features identifiers. The error computation depends on the feature type:
        - For "scalar" features: RMSE normalized by squared reference value.
        - For "field" features: RMSE normalized by field size and max-norm of the reference.

        Args:
            dataset_X: A `Dataset` object or a list of samples.
                Input features used for prediction.
            dataset_y: A `Dataset` object or list, optional.
                Ground-truth targets. If `None`, `dataset_X` is used for both input and reference.

        Returns:
            float: A score between `-inf` and `1`. A perfect prediction yields a score of `1.0`.

        Raises:
            ValueError: If an unknown feature type is encountered.
        """
        check_is_fitted(self, "regressor_")
        if dataset_y is None:
            dataset_y = dataset_X
        if isinstance(dataset_X, list):
            dataset_X = Dataset.from_list_of_samples(dataset_X)
        if isinstance(dataset_y, list):
            dataset_y = Dataset.from_list_of_samples(dataset_y)

        dataset_y_pred = self.predict(dataset_X)

        sample_ids = dataset_X.get_sample_ids()

        assert dataset_y.get_sample_ids() == sample_ids

        all_errors = []

        for feat_id in self.out_features_identifiers_:
            feature_type = feat_id["type"]

            reference = dataset_y.get_feature_from_identifier(feat_id)
            prediction = dataset_y_pred.get_feature_from_identifier(feat_id)

            if feature_type == "scalar":
                errors = 0.0
                for id in sample_ids:
                    if reference[id] != 0:
                        error = ((prediction[id] - reference[id]) ** 2) / (
                            reference[id] ** 2
                        )
                    else:
                        error = (prediction[id] - reference[id]) ** 2
                    errors += error
            elif feature_type == "field":  # pragma: no cover
                errors = 0.0
                for id in sample_ids:
                    errors += (np.linalg.norm(prediction[id] - reference[id]) ** 2) / (
                        reference[id].shape[0]
                        * np.linalg.norm(reference[id], ord=np.inf) ** 2
                    )
            else:  # pragma: no cover
                raise (
                    f"No score function implemented for feature type {feat_id['type']}"
                )

            all_errors.append(np.sqrt(errors / len(sample_ids)))

        return 1.0 - sum(all_errors) / len(self.out_features_identifiers_)
