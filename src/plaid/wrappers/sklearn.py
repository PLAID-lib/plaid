# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
"""Wrappers for scikit-learn estimators/transformers to operate on PLAID Dataset objects in pipelines.

This module provides wrappers for scikit-learn estimators and transformers so they can be used seamlessly in scikit-learn Pipelines
with PLAID objects. The wrapped blocks (e.g. PCA, GaussianProcessRegressor, StandardScaler, etc.) take a `plaid.containers.Dataset` as input,
and return a `plaid.containers.Dataset` as output. This allows you to build
scikit-learn Pipelines where all blocks operate on PLAID objects, enabling end-to-end workflows with domain-specific data structures.

Example usage:

    from sklearn.pipeline import Pipeline
    from plaid.wrappers.sklearn import WrappedSklearnTransform, WrappedSklearnRegressor
    from sklearn.decomposition import PCA
    from sklearn.gaussian_process import GaussianProcessRegressor
    from plaid.containers.dataset import Dataset

    # Define your PLAID dataset
    dataset = Dataset(...)

    # Build a pipeline with wrapped sklearn blocks
    pipe = Pipeline([
        ("pca", WrappedSklearnTransform(PCA(n_components=2))),
        ("reg", WrappedSklearnRegressor(GaussianProcessRegressor()))
    ])

    # Fit the pipeline (all steps receive and return Dataset objects)
    pipe.fit(dataset)
    # Predict
    y_pred = pipe.predict(dataset)

All wrapped blocks must accept and return PLAID Dataset objects.

Some inspiration come from [TensorDict](https://pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule).

This module defines the following classes:
`PlaidWrapper`: Base class for scikit-learn estimators and transformers to operate on PLAID objects.
├── `WrappedSklearnTransform`: Wrapper for scikit-learn Transformer blocks.
└── `WrappedSklearnPredictor`: Wrapper for scikit-learn Predictor blocks.
    ├── `WrappedSklearnClassifier`: Wrapper for scikit-learn Classifier blocks.
    └── `WrappedSklearnRegressor`: Wrapper for scikit-learn Regressor blocks.
"""

# %% Imports
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import logging
from copy import copy
from typing import Union

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

from plaid.containers.dataset import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)

# %% Classes

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


class PlaidWrapper(BaseEstimator, MetaEstimatorMixin):
    """Base wrapper for scikit-learn estimators and transformers to operate on PLAID objects.

    This class is not intended to be used directly, but as a base for wrappers that allow scikit-learn blocks
    (such as PCA, StandardScaler, GaussianProcessRegressor, etc.) to be used in sklearn Pipelines with PLAID objects.
    All methods accept and return `plaid.containers.Dataset` objects.
    """

    def __init__(
        self,
        sklearn_block: SklearnBlock,
        fit_only_ones: bool = True,
        in_features: Union[list[str], str] = [],
        out_features: Union[list[str], str] = [],
    ):
        """Wrap a scikit-learn estimator or transformer.

        Args:
            sklearn_block (SklearnBlock): Any scikit-learn transform or predictor (e.g. PCA, StandardScaler, GPRegressor).
            fit_only_ones (bool, optional): If True, the model will only be fitted once. Defaults to True.
            in_features (Union[list[str],str], optional):
                Names of scalars and/or fields to take as input.
                Scalar (resp. field) names should be given as 'scalar::<scalar-name>' (resp. 'field::<field-name>').
                Use 'all' to use all available scalars and fields, or 'scalar::all'/'field::all' for all scalars/fields.
                Defaults to [].
            out_features (Union[list[str],str], optional): Names of scalars and/or fields to take as output, using the same convention as for `in_features`. Defaults to [].
                Additionally, if 'same', 'scalar::same' or 'field::same' is given, it will use as output the same names as for input.
        """
        # TODO: check https://scikit-learn.org/stable/developers/develop.html#instantiation
        self.sklearn_block = sklearn_block
        self.fit_only_ones = fit_only_ones
        self.in_features = copy(in_features)
        self.out_features = copy(out_features)

    def fit(self, dataset: Dataset, **kwargs):
        """Fit the wrapped scikit-learn model on a PLAID dataset.

        Args:
            dataset (Dataset): The dataset to fit the model on.
            kwargs: Additional keyword arguments to pass to the fit method of the sklearn block.

        Returns:
            self: Returns self for chaining.
        """
        if self.fit_only_ones and self.__sklearn_is_fitted__():
            return self

        self._determine_input_output_names(dataset)

        X, y = self._extract_X_y_from_plaid(dataset)
        self.sklearn_block.fit(X, y, **kwargs)

        self._is_fitted = True
        return self

    def _determine_input_output_names(self, dataset: Dataset):
        """Determine the input/output names based on the in_features and out_features."""
        self.in_features = (
            self.in_features
            if isinstance(self.in_features, list)
            else [self.in_features]
        )
        self.out_features = (
            self.out_features
            if isinstance(self.out_features, list)
            else [self.out_features]
        )
        self._determine_scalar_names(dataset)
        self._determine_time_series_names(dataset)
        self._determine_field_names(dataset)

    def _determine_scalar_names(self, dataset: Dataset):
        """Determine the input/output scalar names based on the in_features and out_features."""
        # Input scalars
        if ("all" in self.in_features) or ("scalar::all" in self.in_features):
            self.input_scalars = dataset.get_scalar_names()
        else:
            self.input_scalars = [
                s[8:] for s in self.in_features if s[:8] == "scalar::"
            ]

        # Output scalars
        if ("all" in self.out_features) or ("scalar::all" in self.out_features):
            assert "same" not in self.out_features
            assert "scalar::same" not in self.out_features
            self.output_scalars = dataset.get_scalar_names()
        elif ("same" in self.out_features) or ("scalar::same" in self.out_features):
            self.output_scalars = self.input_scalars
        else:
            self.output_scalars = [
                s[8:] for s in self.out_features if s[:8] == "scalar::"
            ]

    def _determine_time_series_names(self, dataset: Dataset):
        """Determine the input/output time_series names based on the in_features and out_features."""
        # Input time_series
        if ("all" in self.in_features) or ("time_series::all" in self.in_features):
            self.input_time_series = dataset.get_time_series_names()
        else:
            self.input_time_series = [
                s[8:] for s in self.in_features if s[:8] == "time_series::"
            ]

        # Output time_series
        if ("all" in self.out_features) or ("time_series::all" in self.out_features):
            assert "same" not in self.out_features
            assert "time_series::same" not in self.out_features
            self.output_time_series = dataset.get_time_series_names()
        elif ("same" in self.out_features) or (
            "time_series::same" in self.out_features
        ):
            self.output_time_series = self.input_time_series
        else:
            self.output_time_series = [
                s[8:] for s in self.out_features if s[:8] == "time_series::"
            ]

    def _determine_field_names(self, dataset: Dataset):
        """Determine the input/output field names based on the in_features and out_features."""
        # default_time = dataset[dataset.get_sample_ids()[0]].get_time_assignment()
        # default_base = dataset[dataset.get_sample_ids()[0]].get_base_assignment(time=default_time)
        # has_no_default_base = (default_base is None)
        # if has_no_default_base:
        #     default_zone = dataset[dataset.get_sample_ids()[0]].get_zone_assignment(time=default_time)
        # else:
        #     default_zone = dataset[dataset.get_sample_ids()[0]].get_zone_assignment(base_name=default_base, time=default_time)
        # has_no_default_zone = (default_zone is None)

        # Input fields
        if ("all" in self.in_features) or ("field::all" in self.in_features):
            self.input_fields = dataset.get_field_names()
        else:
            self.input_fields = [s[7:] for s in self.in_features if s[:7] == "field::"]

        # Output fields
        if ("all" in self.out_features) or ("field::all" in self.out_features):
            assert "same" not in self.out_features
            assert "field::same" not in self.out_features
            self.output_fields = dataset.get_field_names()
        elif ("same" in self.out_features) or ("field::same" in self.out_features):
            self.output_fields = self.input_fields
        else:
            self.output_fields = [
                s[7:] for s in self.out_features if s[:7] == "field::"
            ]

    def _extract_X_y_from_plaid(
        self, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features (X) and labels (y) from a PLAID dataset according to the input/output keys.

        Args:
            dataset (Dataset): The dataset to extract data from.

        Returns:
            tuple[np.ndarray, np.ndarray]: The extracted features and labels as numpy arrays.
        """
        ### Inputs
        X = [
            dataset.get_scalars_to_tabular([input_scalar_name], as_nparray=True)
            for input_scalar_name in (
                dataset.get_scalar_names()
                if self.input_scalars == "all"
                else self.input_scalars
            )
        ]
        X.extend(
            [
                dataset.get_time_series_to_tabular(
                    [input_time_series_name], as_nparray=True
                )
                for input_time_series_name in self.input_time_series
            ]
        )
        X.extend(
            [
                dataset.get_fields_to_tabular([input_field_name], as_nparray=True)
                for input_field_name in self.input_fields  # TODO: handle names with '/'
            ]
        )
        # Check shapes
        for i_v, v in enumerate(X):
            # Reshape any 3D arrays to 2D, contracting the last two dimensions
            if len(v.shape) >= 3:
                X[i_v] = v.reshape((len(v), -1))
            # Reshape any 1D arrays to 2D, appending a singleton dimension
            if len(v.shape) == 1:
                X[i_v] = v.reshape((-1, 1))
        # print(f"=== In <_extract_X_y_from_plaid> of {self.sklearn_block=}")
        # print(f"{self.input_scalars=}")
        # print(f"{self.input_fields=}")
        # print(f"{self.output_scalars=}")
        # print(f"{self.output_fields=}")
        # print(f"{type(X)=}")
        # print(f"{len(X)=}")
        # Concatenate the input arrays into a 2D numpy array
        X = np.concatenate(X, axis=-1)

        ### Outputs
        y = [
            dataset.get_scalars_to_tabular([output_scalar_name], as_nparray=True)
            for output_scalar_name in (
                dataset.get_scalar_names()
                if self.output_scalars == "all"
                else self.output_scalars
            )
        ]
        y.extend(
            [
                dataset.get_time_series_to_tabular(
                    [output_time_series_name], as_nparray=True
                )
                for output_time_series_name in self.output_time_series
            ]
        )
        y.extend(
            [
                dataset.get_fields_to_tabular([output_field_name], as_nparray=True)
                for output_field_name in self.output_fields  # TODO: handle names with '/'
            ]
        )
        # Check shapes
        for i_v, v in enumerate(y):
            # Reshape any 3D arrays to 2D, contracting the last two dimensions
            if len(v.shape) >= 3:
                y[i_v] = v.reshape((len(v), -1))
            # Reshape any 1D arrays to 2D, appending a singleton dimension
            if len(v.shape) == 1:
                y[i_v] = v.reshape((-1, 1))
        # print(f"{self.input_scalars=}")
        # print(f"{self.input_fields=}")
        # print(f"{self.output_scalars=}")
        # print(f"{self.output_fields=}")
        # print(f"{type(y)=}")
        # print(f"{len(y)=}")
        # Concatenate the output arrays into a 2D numpy array
        if len(y) > 0:
            y = np.concatenate(y, axis=-1)
        else:
            y = None

        return X, y

    def _convert_y_to_plaid(self, y: np.ndarray, dataset: Dataset) -> Dataset:
        """Convert the model's output (numpy array) to a PLAID Dataset, updating the original dataset.

        Args:
            y (np.ndarray): The model's output.
            dataset (Dataset): The original dataset.

        Returns:
            Dataset: The updated PLAID dataset with new scalars/fields.
        """
        # TODO: use https://scikit-learn.org/stable/glossary.html#term-get_feature_names_out avoid overwriting features
        print("=== In <_convert_y_to_plaid>")
        if hasattr(self.sklearn_block, "feature_names_in_"):
            print(f"- {self.sklearn_block.feature_names_in_=}")
        else:
            print("- self.sklearn_block.feature_names_in_ not found")
        print(f"- {self.sklearn_block.get_feature_names_out()=}")
        print(f"- {self.output_scalars=}")
        print(f"- {self.output_time_series=}")
        print(f"- {self.output_fields=}")
        print(f"- {dataset.get_scalar_names()=}")
        print(f"- {dataset.get_time_series_names()=}")
        print(f"- {dataset.get_field_names()=}")
        print(f"- {y.shape=}")

        new_dset = Dataset()
        # TODO: define tests to determine if we write new features to scalars, fields, or time series
        if y.ndim == 2 and y.shape[0] == len(
            self.sklearn_block.get_feature_names_out()
        ):
            new_dset.add_tabular_scalars(y, self.sklearn_block.get_feature_names_out())
        elif len(self.output_scalars) > 0:
            new_dset.add_tabular_scalars(y, self.output_scalars)

        # if len(self.output_scalars) > 0:
        #     new_dset.add_tabular_scalars(
        #         y[:, : len(self.output_scalars)], self.output_scalars
        #     )
        # if len(self.output_time_series) > 0:
        #     new_dset.add_tabular_time_series(
        #         y[:, len(self.output_scalars) : len(self.output_scalars) + len(self.output_time_series)],
        #         self.output_time_series
        #     )
        # if len(self.output_fields) > 0:
        #     new_dset.add_tabular_fields(
        #         y[:, len(self.output_scalars) + len(self.output_time_series) :],
        #         self.output_fields
        #     )

        dataset.merge_samples(new_dset)
        print(f"- {dataset.get_scalar_names()=}")
        print(f"- {dataset.get_time_series_names()=}")
        print(f"- {dataset.get_field_names()=}")
        return dataset

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


class WrappedSklearnTransform(PlaidWrapper, TransformerMixin):
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
        X, _ = self._extract_X_y_from_plaid(dataset)
        X_transformed = self.sklearn_block.transform(X)
        return self._convert_y_to_plaid(X_transformed, dataset)

    def inverse_transform(self, dataset: Dataset):
        """Inverse transform the dataset using the wrapped sklearn transformer.

        Args:
            dataset (Dataset): The dataset to inverse transform.

        Returns:
            Dataset: The inverse transformed PLAID dataset.
        """
        # TODO: debug
        X, _ = self._extract_X_y_from_plaid(dataset)
        X_transformed = self.sklearn_block.inverse_transform(X)
        return self._convert_y_to_plaid(X_transformed, dataset)

    ## Already defined by TransformerMixin
    # def fit_transform(self, dataset:Dataset):...


class WrappedSklearnPredictor(PlaidWrapper, MetaEstimatorMixin):
    """Wrapper for scikit-learn Predictor blocks to operate on PLAID objects in a Pipeline.

    This class allows you to use any sklearn predictor (e.g. GaussianProcessRegressor, RandomForestRegressor, etc.) in a Pipeline
    where all steps accept and return PLAID Dataset objects. The predict and fit_predict methods take a Dataset and return a new Dataset.
    """

    def predict(self, dataset: Dataset):
        """Predict the output for the given dataset using the wrapped sklearn predictor.

        Args:
            dataset (Dataset): The dataset to predict.

        Returns:
            Dataset: The predicted PLAID dataset.
        """
        X, _ = self._extract_X_y_from_plaid(dataset)
        y_pred = self.sklearn_block.predict(X)
        return self._convert_y_to_plaid(y_pred, dataset)

    def fit_predict(self, dataset: Dataset):
        """Fit the model to the dataset and predict the output using the wrapped sklearn predictor.

        Args:
            dataset (Dataset): The dataset to fit the model on.

        Returns:
            Dataset: The predicted PLAID dataset.
        """
        self.fit(dataset)
        return self.predict(dataset)


class WrappedSklearnClassifier(WrappedSklearnPredictor, ClassifierMixin):
    """Wrapper for scikit-learn Classifier blocks to operate on PLAID objects in a Pipeline.

    Inherits from WrappedSklearnPredictor and ClassifierMixin.
    """

    pass


class WrappedSklearnRegressor(WrappedSklearnPredictor, RegressorMixin):
    """Wrapper for scikit-learn Regressor blocks to operate on PLAID objects in a Pipeline.

    Inherits from WrappedSklearnPredictor and RegressorMixin.
    """

    pass
