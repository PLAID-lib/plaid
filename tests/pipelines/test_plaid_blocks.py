import numpy as np

from plaid.pipelines.plaid_blocks import (
    PlaidColumnTransformer,
    PlaidTransformedTargetRegressor,
)
from plaid.pipelines.sklearn_block_wrappers import (
    get_2Darray_from_homogeneous_identifiers,
)


class Test_PlaidColumnTransformer:
    def test___init__(self, plaid_column_transformer, wrapped_sklearn_transformer):
        PlaidColumnTransformer()
        PlaidColumnTransformer([("titi", wrapped_sklearn_transformer)])
        PlaidColumnTransformer(
            [("toto", wrapped_sklearn_transformer)],
            [{"type": "time_series", "name": "test_time_series"}],
        )
        plaid_column_transformer

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
    def test___init__(self):
        PlaidTransformedTargetRegressor()
