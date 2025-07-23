import numpy as np
import pytest

from plaid.pipelines.sklearn_block_wrappers import (
    WrappedPlaidSklearnRegressor,
    WrappedPlaidSklearnTransformer,
    get_tabular_from_homogeneous_identifiers,
)


def test_get_tabular_from_homogeneous_identifiers(dataset_with_samples):
    # dataset_with_samples.get_all_features_identifiers()
    scalar_feat_id = {"type": "scalar", "name": "test_scalar"}
    feat_ids = [scalar_feat_id]
    X = get_tabular_from_homogeneous_identifiers(dataset_with_samples, feat_ids)
    assert X.shape == (4, 1)

    feat_ids = [scalar_feat_id, scalar_feat_id]
    X = get_tabular_from_homogeneous_identifiers(dataset_with_samples, feat_ids)
    assert X.shape == (4, 2)

    feat_ids = [{"type": "time_series", "name": "test_time_series_1"}]
    X = get_tabular_from_homogeneous_identifiers(dataset_with_samples, feat_ids)
    assert X.shape == (4, 2, 111)

    field_same_size_feat_id = {
        "type": "field",
        "name": "test_field_same_size",
        "base_name": "TestBaseName",
        "zone_name": "TestZoneName",
        "location": "Vertex",
        "time": 0.0,
    }
    feat_ids = [field_same_size_feat_id]
    X = get_tabular_from_homogeneous_identifiers(dataset_with_samples, feat_ids)
    assert X.shape == (4, 17)

    with pytest.raises(ValueError):
        feat_ids = [field_same_size_feat_id, field_same_size_feat_id]
        X = get_tabular_from_homogeneous_identifiers(dataset_with_samples, feat_ids)

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
        X = get_tabular_from_homogeneous_identifiers(dataset_with_samples, feat_ids)


class Test_WrappedPlaidSklearnTransformer:
    def test___init__(self, sklearn_scaler):
        WrappedPlaidSklearnTransformer()
        WrappedPlaidSklearnTransformer(sklearn_block=sklearn_scaler)
        in_feat_ids = [{"type": "scalar", "name": "test_scalar"}]
        WrappedPlaidSklearnTransformer(
            sklearn_block=sklearn_scaler, in_features_identifiers=in_feat_ids
        )
        WrappedPlaidSklearnTransformer(
            sklearn_block=sklearn_scaler,
            in_features_identifiers=in_feat_ids,
            out_features_identifiers=in_feat_ids,
        )

    def test_fit(self, sklearn_scaler, dataset_with_samples):
        in_feat_ids = [{"type": "scalar", "name": "test_scalar"}]
        wrapped_transf = WrappedPlaidSklearnTransformer(
            sklearn_scaler, in_features_identifiers=in_feat_ids
        )
        wrapped_transf.fit(dataset_with_samples)

        wrapped_transf = WrappedPlaidSklearnTransformer(
            sklearn_scaler,
            in_features_identifiers=in_feat_ids,
            out_features_identifiers=in_feat_ids,
        )
        wrapped_transf.fit(dataset_with_samples)

    def test_transform(self, sklearn_scaler, dataset_with_samples):
        in_feat_ids = [{"type": "scalar", "name": "test_scalar"}]
        wrapped_transf = WrappedPlaidSklearnTransformer(
            sklearn_scaler, in_features_identifiers=in_feat_ids
        )
        transformed_dataset = wrapped_transf.fit_transform(dataset_with_samples)
        assert id(dataset_with_samples) != id(transformed_dataset)

        in_features = get_tabular_from_homogeneous_identifiers(
            dataset_with_samples, in_feat_ids
        )
        tranformed_in_features = get_tabular_from_homogeneous_identifiers(
            transformed_dataset, in_feat_ids
        )
        assert not np.allclose(in_features, tranformed_in_features)

    def test_inverse_transform(self, sklearn_scaler, dataset_with_samples):
        in_feat_ids = [{"type": "scalar", "name": "test_scalar"}]
        wrapped_transf = WrappedPlaidSklearnTransformer(
            sklearn_scaler, in_features_identifiers=in_feat_ids
        )
        wrapped_transf.fit(dataset_with_samples)
        transformed_dataset = wrapped_transf.inverse_transform(dataset_with_samples)
        assert id(dataset_with_samples) != id(transformed_dataset)

        in_features = get_tabular_from_homogeneous_identifiers(
            dataset_with_samples, in_feat_ids
        )
        tranformed_in_features = get_tabular_from_homogeneous_identifiers(
            transformed_dataset, in_feat_ids
        )
        assert not np.allclose(in_features, tranformed_in_features)


class Test_WrappedPlaidSklearnRegressor:
    def test___init__(self):
        WrappedPlaidSklearnRegressor()
