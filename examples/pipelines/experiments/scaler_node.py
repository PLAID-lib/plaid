from sklearn.preprocessing import StandardScaler
from persistent_node import PersistentNode


class ScalerNode(PersistentNode):
    def __init__(self, save_path="models/scaler.joblib"):
        super().__init__(save_path)
        self.model = StandardScaler()

    def __call__(self, X):
        return self.fit_transform(X)

    def fit(self, X, y=None):
        if self.load_if_cached(X):
            return self
        self.model.fit(X)
        self.save_model_with_hash(self.model, X)
        return self

    def transform(self, X):
        self.check_fitted_or_load()
        return self.model.transform(X)

    def fit_transform(self, X, y=None):
        if self.load_if_cached(X):
            return self.model.transform(X)
        self.model.fit(X)
        self.save_model_with_hash(self.model, X)
        return self.model.transform(X)

    def inverse_transform(self, X_scaled):
        self.check_fitted_or_load()
        return self.model.inverse_transform(X_scaled)