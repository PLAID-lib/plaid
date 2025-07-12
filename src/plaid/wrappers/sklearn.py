# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#
"""This module provides wrappers for scikit-learn estimators and transformers so they can be used seamlessly in scikit-learn Pipelines
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
        in_keys: Union[list[str], str] = [],
        out_keys: Union[list[str], str] = [],
    ):
        """Wrap a scikit-learn estimator or transformer.

        Args:
            sklearn_block (SklearnBlock): Any scikit-learn transform or predictor (e.g. PCA, StandardScaler, GPRegressor).
            fit_only_ones (bool, optional): If True, the model will only be fitted once. Defaults to True.
            in_keys (Union[list[str],str], optional):
                Names of scalars and/or fields to take as input.
                Scalar (resp. field) names should be given as 'scalar::<scalar-name>' (resp. 'field::<field-name>').
                Use 'all' to use all available scalars and fields, or 'scalar::all'/'field::all' for all scalars/fields.
                Defaults to [].
            out_keys (Union[list[str],str], optional): Names of scalars and/or fields to take as output, using the same convention as for `in_keys`. Defaults to [].
                Additionally, if 'same', 'scalar::same' or 'field::same' is given, it will use as output the same names as for input.
        """
        self.sklearn_block = sklearn_block
        self.fit_only_ones = fit_only_ones
        self.in_keys = in_keys
        self.out_keys = out_keys

        # ---# Scalars
        if in_keys == "all" or "scalar::all" in in_keys:
            self.input_scalars = "all"
        else:
            self.input_scalars = [s[8:] for s in in_keys if s[:8] == "scalar::"]
        #
        if out_keys == "same" or "scalar::same" in out_keys:
            self.output_scalars = self.input_scalars
        else:
            self.output_scalars = [s[8:] for s in out_keys if s[:8] == "scalar::"]

        # ---# Fields
        if in_keys == "all" or "field::all" in in_keys:
            self.input_fields = "all"
        else:
            self.input_fields = [s[7:] for s in in_keys if s[:7] == "field::"]
        #
        if out_keys == "same" or "field::same" in out_keys:
            self.output_fields = self.input_fields
        else:
            self.output_fields = [s[7:] for s in out_keys if s[:7] == "field::"]

        print(f"{self.input_scalars=}")
        print(f"{self.input_fields=}")
        print(f"{self.output_scalars=}")
        print(f"{self.output_fields=}")

    def fit(self, dataset: Dataset, *args, **kwargs):
        """Fit the wrapped scikit-learn model on a PLAID dataset.

        Args:
            dataset (Dataset): The dataset to fit the model on.

        Returns:
            self: Returns self for chaining.
        """
        if self.fit_only_ones and self.__sklearn_is_fitted__():
            return self

        X, y = self._extract_X_y_from_plaid(dataset)
        self.sklearn_block.fit(X, y)

        self._is_fitted = True
        return self

    def _extract_X_y_from_plaid(
        self, dataset: Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features (X) and labels (y) from a PLAID dataset according to the input/output keys.

        Args:
            dataset (Dataset): The dataset to extract data from.

        Returns:
            tuple[np.ndarray, np.ndarray]: The extracted features and labels as numpy arrays.
        """
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
                dataset.get_fields_to_tabular([input_field_name], as_nparray=True)
                for input_field_name in (
                    dataset.get_field_names()
                    if self.input_fields == "all"
                    else self.input_fields
                )
            ]
        )
        # Reshape any 3D arrays to 2D, contracting the last two dimensions
        for i_v, v in enumerate(X):
            if len(v.shape) >= 3:
                X[i_v] = v.reshape((len(v), -1))
            # Reshape any 1D arrays to 2D, appending a singleton dimension
            if len(v.shape) == 1:
                X[i_v] = v.reshape((-1, 1))
        print(f"=== In <_extract_X_y_from_plaid> of {self.sklearn_block=}")
        print(f"{self.input_scalars=}")
        print(f"{self.input_fields=}")
        print(f"{self.output_scalars=}")
        print(f"{self.output_fields=}")
        print(f"{type(X)=}")
        print(f"{len(X)=}")
        # Concatenate the input arrays into a 2D numpy array
        X = np.concatenate(X, axis=-1)

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
                dataset.get_fields_to_tabular([output_field_name], as_nparray=True)
                for output_field_name in (
                    dataset.get_field_names()
                    if self.output_fields == "all"
                    else self.output_fields
                )
            ]
        )
        for i_v, v in enumerate(y):
            # Reshape any 3D arrays to 2D, contracting the last two dimensions
            if len(v.shape) >= 3:
                y[i_v] = v.reshape((len(v), -1))
            # Reshape any 1D arrays to 2D, appending a singleton dimension
            if len(v.shape) == 1:
                y[i_v] = v.reshape((-1, 1))
        # Concatenate the output arrays into a 2D numpy array
        print(f"{self.input_scalars=}")
        print(f"{self.input_fields=}")
        print(f"{self.output_scalars=}")
        print(f"{self.output_fields=}")
        print(f"{type(y)=}")
        print(f"{len(y)=}")
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
        new_dset = Dataset()
        if len(self.output_scalars) > 0:
            new_dset.add_tabular_scalars(
                y[:, : len(self.output_scalars)], self.output_scalars
            )
        if len(self.output_fields) > 0:
            new_dset.add_tabular_fields(
                y[:, len(self.output_scalars) :], self.output_fields
            )
        dataset.merge_samples(new_dset)
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
