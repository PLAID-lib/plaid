"""This file defines shared pytest fixtures and test configurations for pipelines."""

import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


@pytest.fixture()
def sklearn_scaler():
    return MinMaxScaler()


@pytest.fixture()
def sklearn_pca():
    return PCA()


@pytest.fixture()
def sklearn_pca_3comp():
    return PCA(n_components=3)


# @pytest.fixture()
# def wrapped_sklearn_transformer(dataset_with_samples_with_tree: Dataset) -> WrappedPlaidSklearnTransformer:


# dataset_with_samples_with_tree

# wrapped_transf = WrappedPlaidSklearnTransformer(MinMaxScaler(), **config['input_scalar_scaler']['plaid_params'])
