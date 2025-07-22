# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# TODO:
# - recoder score avec la logique
#       def score(self, X, y):
#       y_pred = self.predict(X)
#       return -mean_squared_error(y, y_pred)
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
        sklearn_block: SklearnBlock,
        params:dict
    ):
        assert isinstance(sklearn_block, SklearnBlock), "sklearn_block must be a scikit_learn block"
        assert isinstance(params, dict), "params must be a dict containing the scikit_learn block parameters to set"
        assert "in_features_identifiers" in params, "Must provide in_features_identifiers in sklearn_params of the node"

        self.sklearn_block = sklearn_block
        self.params = params

        # print("kwargs =", kwargs)
        # 1./0.
        # print(self.in_features_identifiers)
        # print(self.in_features_identifiersAA)
        # AttributeError

        self.in_features_identifiers = params['in_features_identifiers']
        check_features_type_homogeneity(self.in_features_identifiers)

        if "out_features_identifiers" in params:
            self.out_features_identifiers = params['out_features_identifiers']
            check_features_type_homogeneity(self.out_features_identifiers)
        else:
            self.out_features_identifiers = params['in_features_identifiers']

        if 'sklearn_params' not in params:
            params['sklearn_params'] = {}

        self.sklearn_block.set_params(**params['sklearn_params'])

    # def _get_param_names(self):
    #     return self.sklearn_block._get_param_names()

    # def get_params(self, deep=True):
    #     return self.sklearn_block.get_params()

    # def set_params(self, **params):
    #     self.params.update(params)
    #     self.sklearn_block.set_params(**params)
    #     return self

    # def get_params(self, deep=True):
    #     out = {
    #         "sklearn_block": self.sklearn_block,
    #     }
    #     out.update(self.params['sklearn_params'])
    #     return out

    # def set_params(self, **params):
    #     if "sklearn_block" in params:
    #         setattr(self, "sklearn_block", params.pop("sklearn_block"))

    #     self.params.update(params)
    #     self.sklearn_block.set_params(**params)

    #     return self

    # def get_params(self, deep=True):
    #     return {
    #         "sklearn_block": self.sklearn_block,
    #         "params": self.params
    #     }

    # def set_params(self, **params):
    #     # Handle sklearn_block and params separately
    #     self.sklearn_block = params.pop("sklearn_block")
    #     self.params = params.pop("params")
    #     # Forward the rest to sklearn_block if any
    #     self.sklearn_block.set_params(**params)
    #     return self


class WrappedPlaidSklearnTransformer(PlaidSklearnBlockWrapper, TransformerMixin):
    """Wrapper for scikit-learn Transformer blocks to operate on PLAID objects in a Pipeline.

    This class allows you to use any sklearn Transformer (e.g. PCA, StandardScaler) in a Pipeline where all steps
    accept and return PLAID Dataset objects. The transform and inverse_transform methods take a Dataset and return a new Dataset.
    """

    def fit(self, dataset: Dataset, y=None):

        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

        X = dataset.get_tabular_from_homogeneous_identifiers(self.in_features_identifiers)
        assert X.shape[1] == len(self.in_features_identifiers), "number of features not consistent between generated tabular and in_features_identifiers"
        X = np.squeeze(X)

        self.sklearn_block_ = clone(self.sklearn_block).fit(X, y)

        return self


    def transform(self, dataset: Dataset):
        """Transform the dataset using the wrapped sklearn transformer.

        Args:
            dataset (Dataset): The dataset to transform.

        Returns:
            Dataset: The transformed PLAID dataset.
        """
        check_is_fitted(self, "sklearn_block_")

        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

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

        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

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

        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

        X = dataset.get_tabular_from_stacked_identifiers(self.in_features_identifiers)
        y = dataset.get_tabular_from_stacked_identifiers(self.out_features_identifiers)

        self.sklearn_block_ = clone(self.sklearn_block).fit(X, y)

        return self

    def predict(self, dataset: Dataset):
        check_is_fitted(self, "sklearn_block_")

        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)

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
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        self.transformer_ = clone(self.transformer).fit(dataset)
        transformed_dataset = self.transformer_.transform(dataset)
        self.regressor_ = clone(self.regressor).fit(transformed_dataset)
        return self

    def predict(self, dataset):
        check_is_fitted(self, "regressor_")
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        dataset_pred_transformed = self.regressor_.predict(dataset)
        return self.transformer_.inverse_transform(dataset_pred_transformed)

    def score(self, dataset_ref, dataset_pred = None):
        check_is_fitted(self, "regressor_")
        if isinstance(dataset_ref, list):
            dataset_ref = Dataset.from_list_of_samples(dataset_ref)
        if isinstance(dataset_pred, list):
            dataset_pred = Dataset.from_list_of_samples(dataset_pred)
        if dataset_pred == None:
            dataset_pred = dataset_ref

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
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        self.transformers_ = []
        for name, transformer, feat_ids in self.transformers:
            sub_dataset = dataset.from_features_identifier(feat_ids)
            transformer_ = clone(transformer).fit(sub_dataset)
            self.transformers_.append((name, transformer_, feat_ids))
        return self

    def transform(self, dataset):
        check_is_fitted(self, "transformers_")
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        dataset_remainder = dataset.from_features_identifier(self.remainder_feature_id)
        transformed_datasets = [dataset_remainder]
        for _, transformer_, feat_ids in self.transformers_:
            sub_dataset = dataset.from_features_identifier(feat_ids)
            transformed = transformer_.transform(sub_dataset)
            transformed_datasets.append(transformed)
        return Dataset.merge_dataset_by_features(transformed_datasets)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


from sklearn.gaussian_process import GaussianProcessRegressor
import inspect
class LazyGPR(BaseEstimator, RegressorMixin):
    def __init__(self, kernel_factory=None, gpr_params=None):
        """
        kernel_factory: Callable[[int], sklearn.gaussian_process.kernels.Kernel]
            Function taking n_features and returning a kernel instance.
        gpr_params: dict
            Parameters passed to GaussianProcessRegressor constructor.
        """
        self.kernel_factory = kernel_factory
        self.gpr_params = gpr_params or {}
        self.gpr_ = None

    def fit(self, X, y):
        if self.kernel_factory is None:
            raise ValueError("kernel_factory must be provided")

        n_features = X.shape[1]
        kernel = self.kernel_factory(n_features)

        self.gpr_ = GaussianProcessRegressor(kernel=kernel, **self.gpr_params)
        self.gpr_.fit(X, y)
        return self

    def predict(self, X, return_std=False, return_cov=False):
        return self.gpr_.predict(X, return_std=return_std, return_cov=return_cov)

    def get_params(self, deep=True):
        # Flatten gpr_params with prefix for better integration with nested wrappers
        params = {"kernel_factory": self.kernel_factory}
        for k, v in self.gpr_params.items():
            params[f"gpr_params__{k}"] = v
        return params

    def set_params(self, **params):
        gpr_params = {}
        for key, value in params.items():
            if key.startswith("gpr_params__"):
                gpr_params[key[len("gpr_params__"):]] = value
            elif key == "kernel_factory":
                self.kernel_factory = value
            else:
                # Unknown param, ignore or raise error
                pass
        if gpr_params:
            self.gpr_params.update(gpr_params)
        return self

    @property
    def kernel_(self):
        return self.gpr_.kernel_ if self.gpr_ is not None else None