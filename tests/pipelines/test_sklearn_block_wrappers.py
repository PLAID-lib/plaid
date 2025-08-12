import numpy as np
import pytest

from plaid.pipelines.sklearn_block_wrappers import (
    WrappedSklearnRegressor,
    WrappedSklearnTransformer,
    get_2Darray_from_homogeneous_identifiers,
)


def test_get_2Darray_from_homogeneous_identifiers(
    dataset_with_samples,
    dataset_with_samples_scalar1_feat_ids,
    dataset_with_samples_scalar2_feat_ids,
    dataset_with_samples_time_series_feat_ids,
):
    # dataset_with_samples.get_all_features_identifiers()
    X = get_2Darray_from_homogeneous_identifiers(
        dataset_with_samples, dataset_with_samples_scalar1_feat_ids
    )
    assert X.shape == (4, 1)

    feat_ids = (
        dataset_with_samples_scalar1_feat_ids + dataset_with_samples_scalar2_feat_ids
    )
    X = get_2Darray_from_homogeneous_identifiers(dataset_with_samples, feat_ids)
    assert X.shape == (4, 2)

    dataset_with_samples_time_series_feat_ids
    # not working yet for time series
    # X = get_2Darray_from_homogeneous_identifiers(
    #     dataset_with_samples, dataset_with_samples_time_series_feat_ids
    # )

    field_same_size_feat_id = {
        "type": "field",
        "name": "test_field_same_size",
        "base_name": "TestBaseName",
        "zone_name": "TestZoneName",
        "location": "Vertex",
        "time": 0.0,
    }
    feat_ids = [field_same_size_feat_id]
    X = get_2Darray_from_homogeneous_identifiers(dataset_with_samples, feat_ids)
    assert X.shape == (4, 17)

    with pytest.raises(ValueError):
        feat_ids = [field_same_size_feat_id, field_same_size_feat_id]
        X = get_2Darray_from_homogeneous_identifiers(dataset_with_samples, feat_ids)

    with pytest.raises(AssertionError):
        feat_ids = [
            {
                "type": "field",
                "name": "test_field_2785",
                "base_name": "TestBaseName",
                "zone_name": "TestZoneName",
                "location": "Vertex",
                "time": 0.0,
            },
            field_same_size_feat_id,
        ]
        X = get_2Darray_from_homogeneous_identifiers(dataset_with_samples, feat_ids)


def test_get_2Darray_from_homogeneous_identifiers_nodes(
    dataset_with_samples_with_tree, dataset_with_samples_with_tree_nodes_feat_ids
):
    X = get_2Darray_from_homogeneous_identifiers(
        dataset_with_samples_with_tree, dataset_with_samples_with_tree_nodes_feat_ids
    )
    assert X.shape == (4, 10)


class Test_WrappedSklearnTransformer:
    def test___init__(
        self,
        sklearn_scaler,
        dataset_with_samples_scalar1_feat_ids,
        dataset_with_samples_scalar2_feat_ids,
    ):
        WrappedSklearnTransformer()
        WrappedSklearnTransformer(sklearn_block=sklearn_scaler)
        # in_feat_ids = [{"type": "scalar", "name": "test_scalar"}]
        WrappedSklearnTransformer(
            sklearn_block=sklearn_scaler,
            in_features_identifiers=dataset_with_samples_scalar1_feat_ids,
        )
        WrappedSklearnTransformer(
            sklearn_block=sklearn_scaler,
            in_features_identifiers=dataset_with_samples_scalar1_feat_ids,
            out_features_identifiers=dataset_with_samples_scalar2_feat_ids,
        )

    def test_fit(
        self,
        wrapped_sklearn_transformer,
        dataset_with_samples,
        dataset_with_samples_scalar2_feat_ids,
    ):
        wrapped_sklearn_transformer.fit(dataset_with_samples)
        wrapped_sklearn_transformer.out_features_identifiers = (
            dataset_with_samples_scalar2_feat_ids
        )
        wrapped_sklearn_transformer.fit(dataset_with_samples)

    def test_transform(self, wrapped_sklearn_transformer, dataset_with_samples):
        transformed_dataset = wrapped_sklearn_transformer.fit_transform(
            dataset_with_samples
        )
        assert id(dataset_with_samples) != id(transformed_dataset)

        in_features = get_2Darray_from_homogeneous_identifiers(
            dataset_with_samples, wrapped_sklearn_transformer.in_features_identifiers_
        )
        tranformed_out_features = get_2Darray_from_homogeneous_identifiers(
            transformed_dataset, wrapped_sklearn_transformer.out_features_identifiers_
        )
        assert not np.allclose(in_features, tranformed_out_features)

    def test_inverse_transform(self, wrapped_sklearn_transformer, dataset_with_samples):
        wrapped_sklearn_transformer.fit(dataset_with_samples)
        transformed_dataset = wrapped_sklearn_transformer.inverse_transform(
            dataset_with_samples
        )
        assert id(dataset_with_samples) != id(transformed_dataset)

        in_features = get_2Darray_from_homogeneous_identifiers(
            dataset_with_samples, wrapped_sklearn_transformer.in_features_identifiers
        )
        tranformed_in_features = get_2Darray_from_homogeneous_identifiers(
            transformed_dataset, wrapped_sklearn_transformer.in_features_identifiers
        )
        assert not np.allclose(in_features, tranformed_in_features)


class Test_WrappedSklearnRegressor:
    def test___init__(
        self,
        sklearn_multioutput_gp_regressor,
        dataset_with_samples_scalar1_feat_ids,
        dataset_with_samples_scalar2_feat_ids,
    ):
        WrappedSklearnRegressor()
        WrappedSklearnRegressor(sklearn_block=sklearn_multioutput_gp_regressor)
        WrappedSklearnRegressor(
            sklearn_block=sklearn_multioutput_gp_regressor,
            in_features_identifiers=dataset_with_samples_scalar1_feat_ids,
        )
        WrappedSklearnRegressor(
            sklearn_block=sklearn_multioutput_gp_regressor,
            in_features_identifiers=dataset_with_samples_scalar1_feat_ids,
            out_features_identifiers=dataset_with_samples_scalar2_feat_ids,
        )

        WrappedSklearnRegressor(
            sklearn_block=sklearn_multioutput_gp_regressor,
            in_features_identifiers=dataset_with_samples_scalar1_feat_ids,
            out_features_identifiers=dataset_with_samples_scalar2_feat_ids,
        )

    def test_fit(self, wrapped_sklearn_multioutput_gp_regressor, dataset_with_samples):
        wrapped_sklearn_multioutput_gp_regressor.fit(dataset_with_samples)

    def test_predict(
        self, wrapped_sklearn_multioutput_gp_regressor, dataset_with_samples
    ):
        out_feat_ids = wrapped_sklearn_multioutput_gp_regressor.out_features_identifiers
        y_ref = get_2Darray_from_homogeneous_identifiers(
            dataset_with_samples, out_feat_ids
        )

        wrapped_sklearn_multioutput_gp_regressor.fit(dataset_with_samples)
        pred_dataset = wrapped_sklearn_multioutput_gp_regressor.predict(
            dataset_with_samples
        )

        assert id(dataset_with_samples) != id(pred_dataset)
        y_pred = get_2Darray_from_homogeneous_identifiers(pred_dataset, out_feat_ids)
        assert np.allclose(y_pred, y_ref)

    def test_transform(self, dataset_with_samples):
        WrappedSklearnRegressor().transform(dataset_with_samples)

    def test_inverse_transform(self, dataset_with_samples):
        WrappedSklearnRegressor().inverse_transform(dataset_with_samples)
