"""This file defines shared pytest fixtures and test configurations for pipelines."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

from plaid.pipelines.plaid_blocks import (
    PlaidColumnTransformer,
)
from plaid.pipelines.sklearn_block_wrappers import (
    WrappedPlaidSklearnRegressor,
    WrappedPlaidSklearnTransformer,
)


@pytest.fixture()
def sklearn_scaler():
    return MinMaxScaler()


@pytest.fixture()
def sklearn_pca():
    return PCA(n_components=3)


@pytest.fixture()
def sklearn_linear_regressor():
    return LinearRegression()


@pytest.fixture()
def sklearn_multioutput_gp_regressor():
    gpr = GaussianProcessRegressor(kernel=RBF())
    return MultiOutputRegressor(gpr)


@pytest.fixture()
def dataset_with_samples_scalar_feat_ids(dataset_with_samples):
    return dataset_with_samples.get_all_features_identifiers_by_type("scalar")


@pytest.fixture()
def dataset_with_samples_time_series_feat_ids(dataset_with_samples):
    return dataset_with_samples.get_all_features_identifiers_by_type("time_series")


@pytest.fixture()
def dataset_with_samples_with_tree_field_feat_ids(dataset_with_samples_with_tree):
    return dataset_with_samples_with_tree.get_all_features_identifiers_by_type("field")


@pytest.fixture()
def dataset_with_samples_with_tree_nodes_feat_ids(dataset_with_samples_with_tree):
    return dataset_with_samples_with_tree.get_all_features_identifiers_by_type("nodes")


# ---------------------------------------------------------------------------------------


@pytest.fixture()
def wrapped_sklearn_transformer(sklearn_scaler, dataset_with_samples_scalar_feat_ids):
    return WrappedPlaidSklearnTransformer(
        sklearn_block=sklearn_scaler,
        in_features_identifiers=dataset_with_samples_scalar_feat_ids,
    )


@pytest.fixture()
def wrapped_sklearn_transformer_2(
    sklearn_pca, dataset_with_samples_time_series_feat_ids
):
    return WrappedPlaidSklearnTransformer(
        sklearn_block=sklearn_pca,
        in_features_identifiers=dataset_with_samples_time_series_feat_ids,
    )


@pytest.fixture()
def wrapped_sklearn_multioutput_gp_regressor(
    sklearn_multioutput_gp_regressor, dataset_with_samples_scalar_feat_ids
):
    def length_scale_init(X):
        return np.ones(X.shape[1])

    dynamics_params_factory = {"estimator__kernel__length_scale": length_scale_init}
    return WrappedPlaidSklearnRegressor(
        sklearn_block=sklearn_multioutput_gp_regressor,
        in_features_identifiers=dataset_with_samples_scalar_feat_ids,
        out_features_identifiers=dataset_with_samples_scalar_feat_ids,
        dynamics_params_factory=dynamics_params_factory,
    )


@pytest.fixture()
def wrapped_sklearn_blocks(
    wrapped_sklearn_transformer, wrapped_sklearn_multioutput_gp_regressor
):
    return [wrapped_sklearn_transformer, wrapped_sklearn_multioutput_gp_regressor]


# ---------------------------------------------------------------------------------------


@pytest.fixture()
def plaid_column_transformer(
    wrapped_sklearn_transformer, wrapped_sklearn_transformer_2
):
    return PlaidColumnTransformer(
        plaid_transformers=[
            ("scalar_scaler", wrapped_sklearn_transformer),
            ("time_series_scaler", wrapped_sklearn_transformer_2),
        ],
        remainder_feature_id=None,
    )


@pytest.fixture()
def plaid_blocks(plaid_column_transformer):
    return [plaid_column_transformer]


# ---------------------------------------------------------------------------------------
@pytest.fixture()
def all_blocks(wrapped_sklearn_blocks, plaid_blocks):
    return wrapped_sklearn_blocks + plaid_blocks


# @pytest.fixture()
# def wrapped_sklearn_transformer(dataset_with_samples_with_tree: Dataset) -> WrappedPlaidSklearnTransformer:


# dataset_with_samples_with_tree

# wrapped_transf = WrappedPlaidSklearnTransformer(MinMaxScaler(), **config['input_scalar_scaler']['plaid_params'])
