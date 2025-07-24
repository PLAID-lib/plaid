import numpy as np

from plaid.pipelines.plaid_blocks import (
    PlaidColumnTransformer,
    PlaidTransformedTargetRegressor,
)
from plaid.pipelines.sklearn_block_wrappers import (
    get_2Darray_from_homogeneous_identifiers,
)


class Test_PlaidColumnTransformer:
    def test___init__(self, wrapped_sklearn_transformer, wrapped_sklearn_transformer_2):
        PlaidColumnTransformer()
        PlaidColumnTransformer([("titi", wrapped_sklearn_transformer)])
        PlaidColumnTransformer(
            [("toto", wrapped_sklearn_transformer)],
            [{"type": "time_series", "name": "test_time_series"}],
        )
        PlaidColumnTransformer(
            plaid_transformers=[
                ("scaler_scalar", wrapped_sklearn_transformer),
                ("pca_field", wrapped_sklearn_transformer_2),
            ],
            remainder_feature_ids=[
                {"type": "time_series", "name": "test_time_series_1"}
            ],
        )

    def test_fit_transform(self, plaid_column_transformer, dataset_with_samples):
        transformed_dataset = plaid_column_transformer.fit_transform(
            dataset_with_samples
        )
        assert id(dataset_with_samples) != id(transformed_dataset)

        in_feat_id_0 = plaid_column_transformer.transformers_[0][
            1
        ].in_features_identifiers_
        in_feat_id_1 = plaid_column_transformer.transformers_[1][
            1
        ].in_features_identifiers_
        out_feat_id_0 = plaid_column_transformer.transformers_[0][
            1
        ].out_features_identifiers_
        out_feat_id_1 = plaid_column_transformer.transformers_[1][
            1
        ].out_features_identifiers_
        in_features_0 = get_2Darray_from_homogeneous_identifiers(
            dataset_with_samples, in_feat_id_0
        )
        in_features_1 = get_2Darray_from_homogeneous_identifiers(
            dataset_with_samples, in_feat_id_1
        )
        out_features_0 = get_2Darray_from_homogeneous_identifiers(
            transformed_dataset, out_feat_id_0
        )
        out_features_1 = get_2Darray_from_homogeneous_identifiers(
            transformed_dataset, out_feat_id_1
        )
        assert not np.allclose(in_features_0, out_features_0)
        assert in_features_1.shape != out_features_1.shape

        transformed_dataset = plaid_column_transformer.fit_transform(
            [s for s in dataset_with_samples]
        )


class Test_PlaidTransformedTargetRegressor:
    def test___init__(
        self,
        wrapped_sklearn_multioutput_gp_regressor,
        wrapped_sklearn_transformer,
        dataset_with_samples_scalar2_feat_ids,
    ):
        PlaidTransformedTargetRegressor()
        PlaidTransformedTargetRegressor(
            regressor=wrapped_sklearn_multioutput_gp_regressor,
        )
        PlaidTransformedTargetRegressor(
            regressor=wrapped_sklearn_multioutput_gp_regressor,
            transformer=wrapped_sklearn_transformer,
        )
        PlaidTransformedTargetRegressor(
            regressor=wrapped_sklearn_multioutput_gp_regressor,
            transformer=wrapped_sklearn_transformer,
            transformed_target_feature_id=dataset_with_samples_scalar2_feat_ids,
        )

    def test_fit(self, plaid_transformed_target_regressor, dataset_with_samples):
        plaid_transformed_target_regressor.fit(dataset_with_samples)
        plaid_transformed_target_regressor.fit([s for s in dataset_with_samples])

    def test_predict(self, plaid_transformed_target_regressor, dataset_with_samples):
        out_feat_ids = plaid_transformed_target_regressor.transformed_target_feature_id
        y_ref = get_2Darray_from_homogeneous_identifiers(
            dataset_with_samples, out_feat_ids
        )

        plaid_transformed_target_regressor.fit(dataset_with_samples)
        pred_dataset = plaid_transformed_target_regressor.predict(dataset_with_samples)

        assert id(dataset_with_samples) != id(pred_dataset)
        y_pred = get_2Darray_from_homogeneous_identifiers(pred_dataset, out_feat_ids)
        assert np.allclose(y_pred, y_ref)
        plaid_transformed_target_regressor.predict([s for s in dataset_with_samples])

    def test_score(self, plaid_transformed_target_regressor, dataset_with_samples):
        plaid_transformed_target_regressor.fit(dataset_with_samples)
        plaid_transformed_target_regressor.score([s for s in dataset_with_samples])
        plaid_transformed_target_regressor.score(
            dataset_with_samples, [s for s in dataset_with_samples]
        )
        score = plaid_transformed_target_regressor.score(dataset_with_samples)
        assert np.isclose(score, 1.0)
