# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# TODO:
# - better handling of params (in the GPRegressor here, params of the GP are not reachable with pipeline.get_params())
# - save state / load state -> normalement ici, tout hérite de BaseEstimator, donc facile ?
# - move this in src and provide tests, examples and move the notebook to the doc


import numpy as np
from sklearn.base import (
    BaseEstimator,
    BiclusterMixin,
    ClassifierMixin,
    ClusterMixin,
    DensityMixin,
    MetaEstimatorMixin,
    MultiOutputMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils._repr_html.estimator import _VisualBlock
from sklearn.utils._tags import get_tags
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from plaid.containers.dataset import Dataset
from plaid.containers.utils import check_features_type_homogeneity
import copy
from typing import Union, Callable, Any
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
"""Union type for all scikit-learn blocks that can be used in a Pipeline."""


available_scalers = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}


class PlaidSklearnBlockWrapper(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        sklearn_block: SklearnBlock = None,
        sklearn_block_constructor: Callable = None,
        params: dict = None,
    ):
        if sklearn_block is None and sklearn_block_constructor is None:
            raise ValueError("You must provide either 'sklearn_block' or 'sklearn_block_constructor'.")
        if sklearn_block is not None and sklearn_block_constructor is not None:
            raise ValueError("Only one of 'sklearn_block' or 'sklearn_block_constructor' should be provided.")
        if sklearn_block is not None:
            assert isinstance(sklearn_block, SklearnBlock), "sklearn_block must by a scikit_learn block"
        if sklearn_block_constructor is not None:
            assert isinstance(sklearn_block_constructor, Callable), "sklearn_block_constructor must by a Callable, returning a scikit_learn block"

        self.sklearn_block = sklearn_block
        self.sklearn_block_constructor = sklearn_block_constructor
        self.params = params or {}

        self.in_features_identifiers = params['in_features_identifiers']
        check_features_type_homogeneity(self.in_features_identifiers)

        if "out_features_identifiers" in params:
            self.out_features_identifiers = params['out_features_identifiers']
            check_features_type_homogeneity(self.out_features_identifiers)
        else:
            self.out_features_identifiers = params['in_features_identifiers']

    def fit_sklearn_block(self, X, y):
        if self.sklearn_block is None:
            self.sklearn_block = self.sklearn_block_constructor(X, y, self.params)
        self.sklearn_block_ = clone(self.sklearn_block).fit(X, y)

class WrappedPlaidSklearnTransformer(PlaidSklearnBlockWrapper, TransformerMixin):
    """Wrapper for scikit-learn Transformer blocks to operate on PLAID objects in a Pipeline.

    This class allows you to use any sklearn Transformer (e.g. PCA, StandardScaler) in a Pipeline where all steps
    accept and return PLAID Dataset objects. The transform and inverse_transform methods take a Dataset and return a new Dataset.
    """

    def fit(self, dataset: Dataset, y=None):

        X = dataset.get_tabular_from_homogeneous_identifiers(self.in_features_identifiers)
        assert X.shape[1] == len(self.in_features_identifiers), "number of features not consistent between generated tabular and in_features_identifiers"
        X = np.squeeze(X)

        self.fit_sklearn_block(X, y)

        return self


    def transform(self, dataset: Dataset):
        """Transform the dataset using the wrapped sklearn transformer.

        Args:
            dataset (Dataset): The dataset to transform.

        Returns:
            Dataset: The transformed PLAID dataset.
        """
        check_is_fitted(self, "sklearn_block_")

        X = dataset.get_tabular_from_homogeneous_identifiers(self.in_features_identifiers)
        assert X.shape[1] == len(self.in_features_identifiers), "number of features not consistent between generated tabular and in_features_identifiers"
        X = np.squeeze(X)

        X_transformed = self.sklearn_block_.transform(X)
        X_transformed = X_transformed.reshape((len(dataset), len(self.out_features_identifiers), -1))

        dataset_transformed = dataset.from_tabular(X_transformed, self.out_features_identifiers, restrict_to_features = False)

        return dataset_transformed

    def inverse_transform(self, dataset: Dataset):
        """Inverse transform the dataset using the wrapped sklearn transformer.

        Args:
            dataset (Dataset): The dataset to inverse transform.

        Returns:
            Dataset: The inverse transformed PLAID dataset.
        """
        check_is_fitted(self, "sklearn_block_")

        X = dataset.get_tabular_from_homogeneous_identifiers(self.out_features_identifiers)
        assert X.shape[1] == len(self.out_features_identifiers), "number of features not consistent between generated tabular and out_features_identifiers"
        X = np.squeeze(X)

        X_inv_transformed = self.sklearn_block_.inverse_transform(X)
        X_inv_transformed = X_inv_transformed.reshape((len(dataset), len(self.in_features_identifiers), -1))

        dataset_inv_transformed = dataset.from_tabular(X_inv_transformed, self.in_features_identifiers, restrict_to_features = False)

        return dataset_inv_transformed

class WrappedPlaidSklearnRegressor(PlaidSklearnBlockWrapper, RegressorMixin):
    """Wrapper for scikit-learn Regressor blocks to operate on PLAID objects in a Pipeline.

    Inherits from PlaidSklearnBlockWrapper and RegressorMixin.
    """
    def fit(self, dataset: Dataset, y=None):

        X = dataset.get_tabular_from_stacked_identifiers(self.in_features_identifiers)
        y = dataset.get_tabular_from_stacked_identifiers(self.out_features_identifiers)

        self.fit_sklearn_block(X, y)

        return self

    def predict(self, dataset: Dataset):
        check_is_fitted(self, "sklearn_block_")

        X = dataset.get_tabular_from_stacked_identifiers(self.in_features_identifiers)

        y = self.sklearn_block_.predict(X)
        y = y.reshape((len(dataset), len(self.out_features_identifiers), -1))

        dataset_predicted = dataset.from_tabular(y, self.out_features_identifiers, restrict_to_features = False)

        return dataset_predicted

    def transform(self, dataset):
        return dataset

    def inverse_transform(self, dataset):
        return dataset




class PlaidTransformedTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, transformer, params = None):
        self.regressor = regressor
        self.transformer = transformer
        self.params = params

    def fit(self, dataset, y=None):
        self.transformer_ = clone(self.transformer).fit(dataset)
        transformed_dataset = self.transformer_.transform(dataset)
        self.regressor_ = clone(self.regressor).fit(transformed_dataset)
        return self

    def predict(self, dataset):
        check_is_fitted(self, "regressor_")
        dataset_pred_transformed = self.regressor_.predict(dataset)
        return self.transformer_.inverse_transform(dataset_pred_transformed)

    def score(self, dataset_ref, dataset_pred):
        check_is_fitted(self, "regressor_")

        sample_ids = dataset_ref.get_sample_ids()

        assert dataset_pred.get_sample_ids() == sample_ids

        in_features_identifiers = self.transformer[0].in_features_identifiers

        scores = []
        for feat_id in in_features_identifiers:

            feature_type = feat_id['type']

            reference  = dataset_ref.get_feature_from_identifier(feat_id)
            prediction = dataset_pred.get_feature_from_identifier(feat_id)

            if feature_type == "scalar":
                errors = 0.
                for id in sample_ids:
                    errors += ((prediction[id] - reference[id])**2)/(reference[id]**2)
            elif feature_type == "field":
                errors = 0.
                for id in sample_ids:
                    errors += (np.linalg.norm(prediction[id] - reference[id])**2)/(reference[id].shape[0]*np.linalg.norm(reference[id], ord = np.inf)**2)
            else:
                raise(f"No score implemented for feature type {feat_id['type']}")

            scores.append(np.sqrt(errors/len(sample_ids)))

        return sum(scores)/len(in_features_identifiers)


class PlaidColumnTransformer(ColumnTransformer):
    def __init__(self,
            transformers: list[tuple[str, Any, list[dict]]],
            remainder_feature_id: list[dict]):
        """
        Parameters
        ----------
        transformers : list of (str, transformer, list[dict])
            List of (name, transformer, features_identifiers).
        remainder_feature_id : list[dict]
            Features to pass through unchanged. The other features are discarded.
        """
        self.transformers = transformers
        self.remainder_feature_id = remainder_feature_id
        super().__init__(transformers)

    def fit(self, dataset, y=None):
        self.transformers_ = []
        for name, transformer, feat_ids in self.transformers:
            sub_dataset = dataset.from_features_identifier(feat_ids)
            transformer_ = clone(transformer).fit(sub_dataset)
            self.transformers_.append((name, transformer_, feat_ids))
        return self

    def transform(self, dataset):
        check_is_fitted(self, "transformers_")
        dataset_remainder = dataset.from_features_identifier(self.remainder_feature_id)
        transformed_datasets = [dataset_remainder]
        for _, transformer_, feat_ids in self.transformers_:
            sub_dataset = dataset.from_features_identifier(feat_ids)
            transformed = transformer_.transform(sub_dataset)
            transformed_datasets.append(transformed)
        return Dataset.merge_dataset_by_features(transformed_datasets)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
