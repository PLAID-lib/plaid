from plaid.pipelines.plaid_blocks import (
    PlaidColumnTransformer,
    PlaidTransformedTargetRegressor,
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


class Test_PlaidTransformedTargetRegressor:
    def test___init__(self):
        PlaidTransformedTargetRegressor()
