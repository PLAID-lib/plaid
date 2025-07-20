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
from plaid.containers.utils import check_features_type_homogeneity
import copy
from typing import Union

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

        # self.in_dim_features = None
        # self.out_dim_features = None

    def fit(self, dataset_in: Dataset, dataset_out: Dataset = None):

        X = dataset_in.get_tabular_from_identifier(self.in_features_identifiers)
        assert X.shape[1] == len(self.in_features_identifiers), "number of features not consistent between generated tabular and in_features_identifiers"
        # self.in_dim_features = X.shape[2]
        X = np.squeeze(X)

        if dataset_out is None:
            y = None
        else:
            y = dataset_out.get_tabular_from_identifier(self.out_features_identifiers)
            assert y.shape[1] == len(self.out_features_identifiers), "number of features not consistent between generated tabular and out_features_identifiers"
            # self.out_dim_features = y.shape[2]
            y = np.squeeze(y)

        self.sklearn_block.fit(X, y)

        self._is_fitted = True
        return self

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


class PlaidSklearnTransformWrapper(PlaidSklearnBlockWrapper, TransformerMixin):
    """Wrapper for scikit-learn Transformer blocks to operate on PLAID objects in a Pipeline.

    This class allows you to use any sklearn Transformer (e.g. PCA, StandardScaler) in a Pipeline where all steps
    accept and return PLAID Dataset objects. The transform and inverse_transform methods take a Dataset and return a new Dataset.
    """

    def transform(self, dataset: Dataset):
        """Transform the dataset using the wrapped sklearn transformer.

        Args:
            dataset (Dataset): The dataset to transform.

        Returns:
            Dataset: The transformed PLAID dataset.
        """
        X = dataset.get_tabular_from_identifier(self.in_features_identifiers)
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
        X = dataset.get_tabular_from_identifier(self.out_features_identifiers)
        assert X.shape[1] == len(self.out_features_identifiers), "number of features not consistent between generated tabular and out_features_identifiers"
        X = np.squeeze(X)

        X_inv_transformed = self.sklearn_block.inverse_transform(X)
        X_inv_transformed = X_inv_transformed.reshape((len(dataset), len(self.in_features_identifiers), -1))

        dataset_inv_transformed = dataset.from_tabular(X_inv_transformed, self.in_features_identifiers, restrict_to_features = False)

        return dataset_inv_transformed

    ## Already defined by TransformerMixin
    # def fit_transform(self, dataset:Dataset):...


class ScalarScalerNode(BaseEstimator, TransformerMixin):

    def __init__(self, params):

        self.params = params

        self.type_ = params['scaler_type']
        assert self.type_ in available_scalers.keys(), "Scaler "+self.type_+" not available"

        self.in_features_identifiers = params['in_features_identifiers']
        if "out_features_identifiers" in params:
            self.in_features_identifiers = params['out_features_identifiers']
        else:
            self.out_features_identifiers = params['in_features_identifiers']

        self.model = None

    def fit(self, dataset, y=None):
        self.model = available_scalers[self.type_]()

        in_features = dataset.get_features_from_identifiers(self.in_features_identifiers)
        X = np.stack(list(in_features.values()))

        self.model.fit(X)
        self.fitted_ = True
        return self

    def transform(self, dataset):
        in_features = dataset.get_features_from_identifiers(self.in_features_identifiers)

        X = np.stack(list(in_features.values()))
        X_transformed = self.model.transform(X)

        out_features_transformed = {time:X_transformed[i,:] for i, time in enumerate(in_features.keys())}

        dataset_transformed = dataset.update_features_from_identifier(self.out_features_identifiers, out_features_transformed, in_place = False)

        return dataset_transformed

    def inverse_transform(self, dataset_transformed):
        out_features_transformed = dataset_transformed.get_features_from_identifiers(self.out_features_identifiers)

        X_transformed = np.stack(list(out_features_transformed.values()))
        X = self.model.inverse_transform(X_transformed)

        in_features = {time:X[i,:] for i, time in enumerate(out_features_transformed.keys())}

        dataset = dataset_transformed.update_features_from_identifier(self.in_features_identifiers, in_features, in_place = False)

        return dataset


class PCAEmbeddingNode(BaseEstimator, RegressorMixin):

    def __init__(self, params, n_components = None):

        self.params = params

        self.n_components = n_components if n_components is not None else params['n_components']

        self.field_name = params['field_name']
        self.zone_name  = params.get('zone_name')
        self.base_name  = params.get('base_name')
        self.location   = params.get('location', 'Vertex')

        self.model = None

    def get_all_fields(self, dataset):
        all_fields = []
        for sample in dataset:
            for time in sample.get_all_mesh_times():
                if self.field_name == "nodes":
                    field = sample.get_nodes(self.zone_name, self.base_name, time).flatten()
                else:
                    field = sample.get_field(self.field_name, self.zone_name, self.base_name, self.location, time)
                all_fields.append(field)
        return np.array(all_fields)

    def set_reduced_fields(self, dataset, reduced_fields):
        for i in dataset.get_sample_ids():
            ii = dataset.get_sample_ids().index(i)
            for j in range(self.n_components):
                dataset[i].add_scalar(f"reduced_{self.field_name}_{j}", reduced_fields[ii, j])

    def get_reduced_fields(self, dataset):
        # if isinstance(dataset, list):
        #     dataset = Dataset.from_list_of_samples(dataset)
        return dataset.get_scalars_to_tabular(
            scalar_names = [f"reduced_{self.field_name}_{j}" for j in range(self.n_components)],
            as_nparray = True
        )

    def set_fields(self, dataset, fields): # TODO: this will not work with multiple times step per sample
        for i in dataset.get_sample_ids():
            ii = dataset.get_sample_ids().index(i)
            for time in dataset[i].get_all_mesh_times():
                dataset[i].add_field(self.field_name, fields[ii], self.zone_name, self.base_name, self.location, time, warning_overwrite = False)

    def fit(self, dataset, y=None):
        self.model = PCA(n_components = self.n_components)

        all_fields = self.get_all_fields(dataset)
        self.model.fit(all_fields)
        self.fitted_ = True
        return self

    def transform(self, dataset):
        all_fields = self.get_all_fields(dataset)
        reduced_fields = self.model.transform(all_fields)
        dataset_ = copy.deepcopy(dataset)
        self.set_reduced_fields(dataset_, reduced_fields)
        return dataset_

    def inverse_transform(self, dataset):
        reduced_fields = self.get_reduced_fields(dataset)
        fields = self.model.inverse_transform(reduced_fields)
        dataset_ = copy.deepcopy(dataset)
        self.set_fields(dataset_, fields)
        return dataset_


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor

available_kernel_classes = {
    "Matern":Matern
}

class GPRegressorNode(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, params):

        self.params = params

        self.type_   = params['type']
        self.input   = params['input']
        self.output  = params['output']
        self.options = params['options']


        assert self.type_ == "GaussianProcessRegressor"
        assert self.options['kernel'] in available_kernel_classes.keys(), "scikit-learn kernel "+self.options['kernel']+" not available"

        self.model = None


    def get_scalars(self, dataset):
        # if isinstance(dataset, list):
        #     dataset = Dataset.from_list_of_samples(dataset)
        return dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )

    def fit(self, dataset, y=None):
        # if isinstance(dataset, list):
        #     dataset = Dataset.from_list_of_samples(dataset)
        all_available_scalar = dataset.get_scalar_names()

        self.input_names = []
        if "scalar_names" in self.input:
            self.input_names += self.input["scalar_names"]
        if "vector_names" in self.input:
            for vn in self.input["vector_names"]:
                self.input_names += [s for s in all_available_scalar if s.startswith(vn)]

        self.output_names = []
        if "scalar_names" in self.output:
            self.output_names += self.output["scalar_names"]
        if "vector_names" in self.output:
            for vn in self.output["vector_names"]:
                self.output_names += [s for s in all_available_scalar if s.startswith(vn)]

        kernel_class = available_kernel_classes[self.options['kernel']]
        if self.options["anisotropic"]:
            kernel = ConstantKernel() * kernel_class(length_scale=np.ones(len(self.input_names)), length_scale_bounds=(1e-8, 1e8),
                                    **self.options["kernel_options"]) + WhiteKernel(noise_level_bounds=(1e-8, 1e8))
        else:
            kernel = kernel_class(length_scale_bounds=(1e-8, 1e8), **self.options["kernel_options"]) \
                + WhiteKernel(noise_level_bounds=(1e-8, 1e8))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=self.options["optim"],
            n_restarts_optimizer=self.options["num_restarts"],
            random_state = self.options["random_state"])

        self.model = MultiOutputRegressor(gpr)
        # if isinstance(dataset, list):
        #     dataset = Dataset.from_list_of_samples(dataset)
        X = dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )
        y = dataset.get_scalars_to_tabular(
            scalar_names = self.output_names,
            as_nparray = True
        )

        self.model.fit(X, y)

        self.fitted_ = True
        return self

    def predict(self, dataset):
        # if isinstance(dataset, list):
        #     dataset = Dataset.from_list_of_samples(dataset)
        X = dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )

        pred = self.model.predict(X)
        if len(self.output_names) == 1:
            pred = pred.reshape((-1, 1))

        dataset_ = copy.deepcopy(dataset)
        for i in dataset.get_sample_ids():
            ii = dataset.get_sample_ids().index(i)
            for j, sn in enumerate(self.output_names):
                dataset_[i].add_scalar(sn, pred[ii, j])

        return dataset_

    def transform(self, dataset):
        return dataset

    def inverse_transform(self, dataset):
        return dataset

    # def score(self, dataset, dataset_ref):
    #     if not dataset_ref:
    #         # case where GirdSearchCV is called with only one argument search.fit(dataset)
    #         dataset_ref = dataset
    #     if isinstance(dataset, list):
    #         dataset = Dataset.from_list_of_samples(dataset)
    #     X = dataset.get_scalars_to_tabular(
    #         scalar_names = self.input_names,
    #         as_nparray = True
    #     )
    #     if isinstance(dataset_ref, list):
    #         dataset_ref = Dataset.from_list_of_samples(dataset_ref)
    #     y = dataset_ref.get_scalars_to_tabular(
    #         scalar_names = self.output_names,
    #         as_nparray = True
    #     )
    #     return self.model.score(X, y)


# class PLAIDColumnTransformer(ColumnTransformer):
#     def __init__(self, transformers, **kwargs):
#         # transformers = [(name, transformer, selector)]
#         # Here, selector can be a callable or a list of identifiers
#         self._raw_transformers = transformers
#         self._plaid_transformers_ = None
#         super().__init__([], **kwargs)

# class PLAIDColumnTransformer(ColumnTransformer):
class PLAIDColumnTransformer(ColumnTransformer):
    def __init__(self, transformers):
        """
        transformers: list of (name, transformer, feature_selector)
        - name: str, label for the transformer
        - transformer: must have fit(), transform(), optionally fit_transform()
        - feature_selector: list of feature identifiers, or callable(dataset) → identifiers
        """
        # self._raw_transformers = transformers  # Store true selectors
        # # Dummy selectors just for nice rendering
        transformers = [
            (name, transformer, f"<{name}_features>")
            for name, transformer in transformers
        ]
        # print(dummy_transformers)
        super().__init__(transformers)

    def fit(self, dataset, y=None):
        self.fitted_transformers_ = []
        for name, transformer, selector in self.transformers:
            if callable(selector):
                identifiers = selector(dataset)
            else:
                identifiers = selector
            features = dataset.get_features_from_identifiers(identifiers)
            transformer.fit(features)
            self.fitted_transformers_.append((name, transformer, identifiers))
        return self

    def transform(self, dataset):
        transformed_blocks = []
        for name, transformer, identifiers in self.fitted_transformers_:
            features = dataset.get_features_from_identifiers(identifiers)
            transformed = transformer.transform(features)
            transformed_blocks.append(transformed)
        return transformed_blocks  # or merge into a dict, or update dataset

    def fit_transform(self, dataset, y=None):
        self.fit(dataset, y)
        return self.transform(dataset)


class PLAIDTransformedTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, transformer):
        self.regressor = regressor
        self.transformer = transformer


    def get_all_fields(self, dataset):
        all_fields = []
        for sample in dataset:
            all_fields.append(sample.get_field("mach", base_name = "Base_2_2"))
        return np.array(all_fields)


    def fit(self, dataset, y=None):
        transformed_dataset = self.transformer.fit_transform(dataset)

        self.regressor.fit(transformed_dataset)
        self.fitted_ = True
        return self

    def predict(self, dataset):
        dataset_pred_transformed = self.regressor.predict(dataset)
        return self.transformer.inverse_transform(dataset_pred_transformed)

    def score(self, dataset_ref, dataset_pred):

        mach_ref = self.get_all_fields(dataset_ref)
        mach_pred = self.get_all_fields(dataset_pred)

        n_samples = len(mach_ref)
        errors = 0
        for i in range(n_samples):
            errors += (np.linalg.norm(mach_pred[i] - mach_ref[i])**2)/(mach_ref[i].shape[0]*np.linalg.norm(mach_ref[i], ord = np.inf)**2)
        score = np.sqrt(errors/n_samples)
        return score



# class PLAIDTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, features_param):

#         self.features_param = features_param

#     def get_features(self, dataset):
#         return dataset.get_features_from_identifiers(self.features_param)

#     def set_features(self, dataset, features):
#         return dataset.update_features_from_identifier(self.features_param, features, in_place = False)

#     def fit(self, dataset, y=None):

#         return self

#     def transform(self, dataset):
#         return dataset

#     def inverse_transform(self, dataset):
#         return dataset
