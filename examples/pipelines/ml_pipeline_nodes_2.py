# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

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
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from plaid.containers.dataset import Dataset
from plaid.containers.utils import check_features_type_homogeneity, merge_dataset_by_features
import copy
from typing import Union, Callable, Any

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
        sklearn_block: SklearnBlock,
        params:dict
    ):
        # TODO: check https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.sklearn_block = sklearn_block
        self.params = params

        self.in_features_identifiers = params['in_features_identifiers']
        check_features_type_homogeneity(self.in_features_identifiers)

        if "out_features_identifiers" in params:
            self.out_features_identifiers = params['out_features_identifiers']
            check_features_type_homogeneity(self.out_features_identifiers)
        else:
            self.out_features_identifiers = params['in_features_identifiers']

    def __sklearn_is_fitted__(self):
        """Check if the wrapped scikit-learn model is fitted.

        Returns:
            bool: True if the model is fitted, False otherwise.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def __repr__(self):
        """String representation of the wrapper, showing the underlying sklearn block."""
        return f"{self.__class__.__name__}({self.sklearn_block.__repr__()})"

    def __str__(self):
        """String representation of the wrapper, showing the underlying sklearn block."""
        return f"{self.__class__.__name__}({self.sklearn_block.__str__()})"


class PlaidSklearnBlockConstructorWrapper(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        sklearn_block_constructor: Callable,
        params:dict
    ):
        # TODO: check https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.sklearn_block_constructor = sklearn_block_constructor
        self.sklearn_block = None

        self.params = params

        self.in_features_identifiers = params['in_features_identifiers']
        check_features_type_homogeneity(self.in_features_identifiers)

        if "out_features_identifiers" in params:
            self.out_features_identifiers = params['out_features_identifiers']
            check_features_type_homogeneity(self.out_features_identifiers)
        else:
            self.out_features_identifiers = params['in_features_identifiers']

    def __sklearn_is_fitted__(self):
        """Check if the wrapped scikit-learn model is fitted.

        Returns:
            bool: True if the model is fitted, False otherwise.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted

    def __repr__(self):
        """String representation of the wrapper, showing the underlying sklearn block."""
        return f"{self.__class__.__name__}({self.sklearn_block.__repr__()})"

    def __str__(self):
        """String representation of the wrapper, showing the underlying sklearn block."""
        return f"{self.__class__.__name__}({self.sklearn_block.__str__()})"


class WrappedPlaidSklearnTransformer(PlaidSklearnBlockWrapper, TransformerMixin):
    """Wrapper for scikit-learn Transformer blocks to operate on PLAID objects in a Pipeline.

    This class allows you to use any sklearn Transformer (e.g. PCA, StandardScaler) in a Pipeline where all steps
    accept and return PLAID Dataset objects. The transform and inverse_transform methods take a Dataset and return a new Dataset.
    """

    def fit(self, dataset: Dataset):

        X = dataset.get_tabular_from_homogeneous_identifiers(self.in_features_identifiers)
        assert X.shape[1] == len(self.in_features_identifiers), "number of features not consistent between generated tabular and in_features_identifiers"
        X = np.squeeze(X)

        self.sklearn_block.fit(X, y = None)

        self._is_fitted = True
        return self


    def transform(self, dataset: Dataset):
        """Transform the dataset using the wrapped sklearn transformer.

        Args:
            dataset (Dataset): The dataset to transform.

        Returns:
            Dataset: The transformed PLAID dataset.
        """
        X = dataset.get_tabular_from_homogeneous_identifiers(self.in_features_identifiers)
        assert X.shape[1] == len(self.in_features_identifiers), "number of features not consistent between generated tabular and in_features_identifiers"
        X = np.squeeze(X)

        X_transformed = self.sklearn_block.transform(X)
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
        X = dataset.get_tabular_from_homogeneous_identifiers(self.out_features_identifiers)
        assert X.shape[1] == len(self.out_features_identifiers), "number of features not consistent between generated tabular and out_features_identifiers"
        X = np.squeeze(X)

        X_inv_transformed = self.sklearn_block.inverse_transform(X)
        X_inv_transformed = X_inv_transformed.reshape((len(dataset), len(self.in_features_identifiers), -1))

        dataset_inv_transformed = dataset.from_tabular(X_inv_transformed, self.in_features_identifiers, restrict_to_features = False)

        return dataset_inv_transformed

class WrappedPlaidSklearnRegressor(PlaidSklearnBlockConstructorWrapper, RegressorMixin):
    """Wrapper for scikit-learn Regressor blocks to operate on PLAID objects in a Pipeline.

    Inherits from PlaidSklearnBlockWrapper and RegressorMixin.
    """
    def fit(self, dataset: Dataset):

        X = dataset.get_tabular_from_stacked_identifiers(self.in_features_identifiers)
        y = dataset.get_tabular_from_stacked_identifiers(self.out_features_identifiers)

        self.sklearn_block = self.sklearn_block_constructor(X)

        self.sklearn_block.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, dataset: Dataset):

        X = dataset.get_tabular_from_stacked_identifiers(self.in_features_identifiers)

        y = self.sklearn_block.predict(X)
        y = y.reshape((len(dataset), len(self.out_features_identifiers), -1))

        dataset_predicted = dataset.from_tabular(y, self.out_features_identifiers, restrict_to_features = False)

        return dataset_predicted

    def transform(self, dataset):
        return dataset

    def inverse_transform(self, dataset):
        return dataset




class PLAIDTransformedTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, transformer):
        self.regressor = regressor
        self.transformer = transformer

    def fit(self, dataset, y=None):
        transformed_dataset = self.transformer.fit_transform(dataset)

        self.regressor.fit(transformed_dataset)
        self.fitted_ = True
        return self

    def predict(self, dataset):
        dataset_pred_transformed = self.regressor.predict(dataset)

        return self.transformer.inverse_transform(dataset_pred_transformed)

    def score(self, dataset_ref, dataset_pred):

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


# class PLAIDFeatureTransformer(ColumnTransformer):
#     def __init__(self, transformers):
#         """
#         transformers: list of (name, transformer, feature_selector)
#         - name: str, label for the transformer
#         - transformer: must have fit(), transform(), optionally fit_transform()
#         - feature_selector: list of feature identifiers, or callable(dataset) → identifiers
#         """
#         # self._raw_transformers = transformers  # Store true selectors
#         # # Dummy selectors just for nice rendering
#         transformers = [
#             (name, transformer, f"<{name}_features>")
#             for name, transformer in transformers
#         ]
#         # print(dummy_transformers)
#         super().__init__(transformers)

#     def fit(self, dataset, y=None):
#         self.fitted_transformers_ = []
#         for name, transformer, selector in self.transformers:
#             if callable(selector):
#                 identifiers = selector(dataset)
#             else:
#                 identifiers = selector
#             features = dataset.get_features_from_identifiers(identifiers)
#             transformer.fit(features)
#             self.fitted_transformers_.append((name, transformer, identifiers))
#         return self

#     def transform(self, dataset):
#         transformed_blocks = []
#         for _, transformer, identifiers in self.fitted_transformers_:
#             features = dataset.get_features_from_identifiers(identifiers)
#             transformed = transformer.transform(features)
#             transformed_blocks.append(transformed)
#         return transformed_blocks  # or merge into a dict, or update dataset

#     def fit_transform(self, dataset, y=None):
#         self.fit(dataset, y)
#         return self.transform(dataset)



class PLAIDFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
            transformers: list[tuple[str, Any, list[dict]]],
            remainder_feature_id: list):
        """
        transformers: List of (name, transformer, features_identifiers)
        remainder_feature_id: List of features_identifiers
        """
        self.transformers = transformers
        self.remainder_feature_id = remainder_feature_id
        print("remainder_feature_id =", remainder_feature_id)

    def fit(self, dataset, y=None):
        for _, transformer, feat_ids in self.transformers:
            sub_dataset = dataset.from_features_identifier(feat_ids)
            transformer.fit(sub_dataset)
        return self

    def transform(self, dataset):
        dataset_remainder = dataset.from_features_identifier(self.remainder_feature_id)
        transformed_datasets = [dataset_remainder]
        for _, transformer, feat_ids in self.transformers:
            sub_dataset = dataset.from_features_identifier(feat_ids)
            transformed = transformer.transform(sub_dataset)
            transformed_datasets.append(transformed)
        merged_dataset = merge_dataset_by_features(transformed_datasets)
        print("merged_dataset =", merged_dataset)
        return merged_dataset

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)