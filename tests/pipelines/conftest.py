"""This file defines shared pytest fixtures and test configurations for pipelines."""

import pytest
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

from plaid.pipelines.plaid_blocks import (
    ColumnTransformer,
    TransformedTargetRegressor,
)
from plaid.pipelines.sklearn_block_wrappers import (
    WrappedSklearnRegressor,
    WrappedSklearnTransformer,
)


@pytest.fixture()
def sklearn_scaler():
    return MinMaxScaler()


@pytest.fixture()
def sklearn_pca():
    return PCA(n_components=2)


@pytest.fixture()
def sklearn_linear_regressor():
    return LinearRegression()


@pytest.fixture()
def sklearn_multioutput_gp_regressor():
    gpr = GaussianProcessRegressor(kernel=RBF(), random_state=42)
    return MultiOutputRegressor(gpr)


@pytest.fixture()
def dataset_with_samples_scalar1_feat_ids(dataset_with_samples):
    return [dataset_with_samples.get_all_features_identifiers_by_type("scalar")[0]]


@pytest.fixture()
def dataset_with_samples_scalar2_feat_ids(dataset_with_samples):
    return [dataset_with_samples.get_all_features_identifiers_by_type("scalar")[1]]


@pytest.fixture()
def dataset_with_samples_time_series_feat_ids(dataset_with_samples):
    return dataset_with_samples.get_all_features_identifiers_by_type("time_series")


@pytest.fixture()
def dataset_with_samples_with_mesh_field_feat_ids(dataset_with_samples_with_mesh):
    return dataset_with_samples_with_mesh.get_all_features_identifiers_by_type("field")


@pytest.fixture()
def dataset_with_samples_with_mesh_1field_feat_ids(dataset_with_samples_with_mesh):
    return [
        dataset_with_samples_with_mesh.get_all_features_identifiers_by_type("field")[0]
    ]


@pytest.fixture()
def dataset_with_samples_with_mesh_nodes_feat_ids(dataset_with_samples_with_mesh):
    return dataset_with_samples_with_mesh.get_all_features_identifiers_by_type("nodes")


# ---------------------------------------------------------------------------------------


@pytest.fixture()
def wrapped_sklearn_transformer(sklearn_scaler, dataset_with_samples_scalar1_feat_ids):
    return WrappedSklearnTransformer(
        sklearn_block=sklearn_scaler,
        in_features_identifiers=dataset_with_samples_scalar1_feat_ids,
    )


@pytest.fixture()
def wrapped_sklearn_transformer_2(sklearn_pca):
    return WrappedSklearnTransformer(
        sklearn_block=sklearn_pca,
        in_features_identifiers=[{"type": "field", "name": "test_field_same_size"}],
    )


@pytest.fixture()
def wrapped_sklearn_multioutput_gp_regressor(
    sklearn_multioutput_gp_regressor,
    dataset_with_samples_scalar1_feat_ids,
    dataset_with_samples_scalar2_feat_ids,
):
    return WrappedSklearnRegressor(
        sklearn_block=sklearn_multioutput_gp_regressor,
        in_features_identifiers=dataset_with_samples_scalar2_feat_ids,
        out_features_identifiers=dataset_with_samples_scalar1_feat_ids,
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
    return ColumnTransformer(
        plaid_transformers=[
            ("scaler_scalar", wrapped_sklearn_transformer),
            ("pca_field", wrapped_sklearn_transformer_2),
        ],
    )


@pytest.fixture()
def plaid_transformed_target_regressor(
    wrapped_sklearn_multioutput_gp_regressor, wrapped_sklearn_transformer
):
    return TransformedTargetRegressor(
        regressor=wrapped_sklearn_multioutput_gp_regressor,
        transformer=wrapped_sklearn_transformer,
    )


@pytest.fixture()
def plaid_blocks(plaid_column_transformer, plaid_transformed_target_regressor):
    return [plaid_column_transformer, plaid_transformed_target_regressor]


# ---------------------------------------------------------------------------------------
@pytest.fixture()
def all_blocks(wrapped_sklearn_blocks, plaid_blocks):
    return wrapped_sklearn_blocks + plaid_blocks
