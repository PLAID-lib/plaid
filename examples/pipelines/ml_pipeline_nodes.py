import os
import json
import joblib
from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from plaid.containers.dataset import Dataset
import copy

from sklearn.preprocessing import StandardScaler, MinMaxScaler

available_scalers = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}

class ScalarScalerNode(BaseEstimator, TransformerMixin):

    def __init__(self, type, scalar_names):
        self.type = type
        self.scalar_names = scalar_names

        assert type in available_scalers.keys(), "Scaler "+type+" not available"

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
        self.model = available_scalers[self.type]()

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

    def __init__(self, field_name = None, n_components = None, zone_name = None, base_name = None, time = None, location = "Vertex"):

        self.zone_name = zone_name
        self.base_name = base_name
        self.time      = time
        self.location  = location

        self.field_name   = field_name
        self.n_components = n_components

        self.model = None

    def get_all_fields(self, dataset):
        all_fields = []
        for sample in dataset:
            if self.field_name == "nodes":
                field = sample.get_nodes(self.zone_name, self.base_name, self.time).flatten()
            else:
                field = sample.get_field(self.field_name, self.zone_name, self.base_name, self.location, self.time)
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

    def set_fields(self, dataset, fields):
        for i in range(len(dataset)):
            dataset[i].add_field(self.field_name, fields[i], self.zone_name, self.base_name, self.location, self.time)

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
        assert params['type'] == "GaussianProcessRegressor"
        assert self.params['options']["kernel"] in available_kernel_classes.keys(), "scikit-learn kernel "+self.params['options']["kernel"]+" not available"

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
        if "scalar_names" in self.params['input']:
            self.input_names += self.params['input']["scalar_names"]
        if "vector_names" in self.params['input']:
            for vn in self.params['input']["vector_names"]:
                self.input_names += [s for s in all_available_scalar if s.startswith(vn)]

        self.output_names = []
        if "scalar_names" in self.params['output']:
            self.output_names += self.params['output']["scalar_names"]
        if "vector_names" in self.params['output']:
            for vn in self.params['output']["vector_names"]:
                self.output_names += [s for s in all_available_scalar if s.startswith(vn)]

        kernel_class = available_kernel_classes[self.params['options']["kernel"]]
        if self.params['options']["anisotropic"]:
            kernel = ConstantKernel() * kernel_class(length_scale=np.ones(len(self.input_names)), length_scale_bounds=(1e-8, 1e8),
                                    **self.params['options']["kernel_options"]) + WhiteKernel(noise_level_bounds=(1e-8, 1e8))
        else:
            kernel = kernel_class(length_scale_bounds=(1e-8, 1e8), **self.params['options']["kernel_options"]) \
                + WhiteKernel(noise_level_bounds=(1e-8, 1e8))

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=self.params['options']["optim"],
            n_restarts_optimizer=self.params['options']["num_restarts"],
            random_state = self.params['options']["random_state"])

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

