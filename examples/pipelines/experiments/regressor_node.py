from sklearn.linear_model import Ridge
from sklearn.base import RegressorMixin
from persistent_node import PersistentNode


class RegressorNode(PersistentNode, RegressorMixin):
    def __init__(self, save_path="models/regressor.joblib"):
        super().__init__(save_path)
        self.model = Ridge()

    def __call__(self, X, y=None):
        if y is not None:
            self.fit(X, y)
        return self.predict(X)

    def fit(self, X, y):
        if self.load_if_cached((X, y)):
            return self
        self.model.fit(X, y)
        self.save_model_with_hash(self.model, (X, y))
        return self

    def predict(self, X):
        self.check_fitted_or_load()
        return self.model.predict(X)

    def inverse_transform(self, y_pred):
        return y_pred
