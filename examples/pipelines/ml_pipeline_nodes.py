# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.decomposition import PCA
from plaid.containers.dataset import Dataset
import copy

from sklearn.preprocessing import StandardScaler, MinMaxScaler

available_scalers = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}

class ScalarScalerNode(BaseEstimator, TransformerMixin):

    def __init__(self, params):

        self.params = params

        self.type_ = params['type']
        self.scalar_names = params['scalar_names']

        assert self.type_ in available_scalers.keys(), "Scaler "+self.type_+" not available"

        self.model = None

    def get_scalars(self, dataset):
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        return dataset.get_scalars_to_tabular(
            scalar_names = self.scalar_names,
            as_nparray = True
        )

    def set_scalars(self, dataset, scalars):
        for i in range(len(dataset)):
            for j, sn in enumerate(self.scalar_names):
                dataset[i].add_scalar(sn, scalars[i, j])

    def fit(self, dataset, y=None):
        self.model = available_scalers[self.type_]()

        scalars = self.get_scalars(dataset)
        self.model.fit(scalars)
        self.fitted_ = True
        return self

    def transform(self, dataset):
        scalars = self.get_scalars(dataset)
        scaled_scalars = self.model.transform(scalars)
        dataset_ = copy.deepcopy(dataset)
        self.set_scalars(dataset_, scaled_scalars)
        return dataset_

    def inverse_transform(self, dataset):
        scaled_scalars = self.get_scalars(dataset)
        scalars = self.model.inverse_transform(scaled_scalars)
        dataset_ = copy.deepcopy(dataset)
        self.set_scalars(dataset_, scalars)
        return dataset_


class PCAEmbeddingNode(BaseEstimator, RegressorMixin, TransformerMixin):

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
        for i in range(len(dataset)):
            for j in range(self.n_components):
                dataset[i].add_scalar(f"reduced_{self.field_name}_{j}", reduced_fields[i, j])

    def get_reduced_fields(self, dataset):
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        return dataset.get_scalars_to_tabular(
            scalar_names = [f"reduced_{self.field_name}_{j}" for j in range(self.n_components)],
            as_nparray = True
        )

    def set_fields(self, dataset, fields): # TODO: this will not work with multiple times step per sample
        for i in range(len(dataset)):
            for time in dataset[i].get_all_mesh_times():
                dataset[i].add_field(self.field_name, fields[i], self.zone_name, self.base_name, self.location, time)

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
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        return dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )

    def fit(self, dataset, y=None):
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
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
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
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
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        X = dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )

        pred= self.model.predict(X)
        if len(self.output_names) == 1:
            pred = pred.reshape((-1, 1))

        dataset_ = copy.deepcopy(dataset)
        for i in range(len(dataset)):
            for j, sn in enumerate(self.output_names):
                dataset_[i].add_scalar(sn, pred[i, j])

        return dataset_

    def transform(self, dataset):
        return dataset

    def inverse_transform(self, dataset):
        return dataset

    def score(self, dataset, dataset_ref):
        if not dataset_ref:
            # case where GirdSearchCV is called with only one argument search.fit(dataset)
            dataset_ref = dataset
        if isinstance(dataset, list):
            dataset = Dataset.from_list_of_samples(dataset)
        X = dataset.get_scalars_to_tabular(
            scalar_names = self.input_names,
            as_nparray = True
        )
        if isinstance(dataset_ref, list):
            dataset_ref = Dataset.from_list_of_samples(dataset_ref)
        y = dataset_ref.get_scalars_to_tabular(
            scalar_names = self.output_names,
            as_nparray = True
        )
        return self.model.score(X, y)

