from plaid.pipelines.plaid_blocks import (
    PlaidColumnTransformer,
    PlaidTransformedTargetRegressor,
)


class Test_PlaidTransformedTargetRegressor:
    def test___init__(self):
        PlaidTransformedTargetRegressor()


class Test_PlaidColumnTransformer:
    def test___init__(self):
        PlaidColumnTransformer()
