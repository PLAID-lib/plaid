import pickle
import time

import numpy as np
from datasets import load_dataset
from GPy.kern import RBF, Matern32, Matern52
from GPy.models import GPRegression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from plaid.containers.sample import Sample

dataset = load_dataset("PLAID-datasets/Rotor37", split="all_samples")

ids_train = dataset.description["split"]["train_1000"]
ids_test = dataset.description["split"]["test"]

out_fields_names = ["Density", "Pressure", "Temperature"]
out_scalars_names = ["Massflow", "Compression_ratio", "Efficiency"]


def convert_data(
    ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Converts a list of sample IDs into structured numpy arrays containing input and output data.

    Parameters:
    ----------
    ids : list[int]
        list of sample indices to retrieve from the dataset.

    Returns:
    -------
    nodes : np.ndarray
        Flattened array of node features for each sample.
    X_scalars : np.ndarray
        Array containing input scalar values (Omega, Pressure) for each sample.
    Y_scalars : np.ndarray
        Array containing output scalar values (Massflow, Compression Ratio, Efficiency).
    Y_density : np.ndarray
        Array containing field values for Density across samples.
    Y_pressure : np.ndarray
        Array containing field values for Pressure across samples.
    Y_temperature : np.ndarray
        Array containing field values for Temperature across samples.
    """
    X_scalars = []
    Y_scalars = []
    Y_density = []
    Y_pressure = []
    Y_temperature = []
    nodes = []

    for id in ids:
        sample = Sample.model_validate(pickle.loads(dataset[id]["sample"]))

        nodes.append(sample.get_nodes())
        omega = sample.scalars.get("Omega")
        pressure = sample.scalars.get("P")

        density = sample.get_field("Density")
        pressure_field = sample.get_field("Pressure")
        temperature = sample.get_field("Temperature")

        massflow = sample.scalars.get("Massflow")
        compression_ratio = sample.scalars.get("Compression_ratio")
        efficiency = sample.scalars.get("Efficiency")

        X_scalars.append(np.array([omega, pressure]))
        Y_scalars.append(np.array([massflow, compression_ratio, efficiency]))
        Y_density.append(density)
        Y_pressure.append(pressure_field)
        Y_temperature.append(temperature)

    # Convert lists to numpy arrays
    nodes = np.stack(nodes).reshape(len(nodes), -1)
    X_scalars = np.stack(X_scalars)
    Y_scalars = np.stack(Y_scalars)
    Y_density = np.stack(Y_density)
    Y_pressure = np.stack(Y_pressure)
    Y_temperature = np.stack(Y_temperature)

    return nodes, X_scalars, Y_scalars, Y_density, Y_pressure, Y_temperature


class GPyRegressor(BaseEstimator, RegressorMixin):
    """Custom Gaussian Process Regressor using GPy library.

    Args:
        normalizer (bool): Whether to normalize the output.
        kernel (str): Type of kernel to use in the Gaussian Process.
            Options: 'Matern52', 'Matern32', 'Rbf'.
        num_restarts (int): Number of restarts for kernel optimization.
    """

    def __init__(
        self, normalizer: bool = False, kernel: str = "Matern52", num_restarts: int = 5
    ):
        self.normalizer = normalizer
        self.kernel = kernel
        self.num_restarts = num_restarts

    def fit(self, X, y):
        """Fit the Gaussian Process model to the data.

        Args:
            X (ndarray): Input features of shape (n_samples, n_features).
            y (ndarray): Target values of shape (n_samples,) or (n_samples, 1).

        Returns:
            self: Returns the instance of the fitted model.
        """
        # Reshape y to have shape (n_samples, 1) if it's 1D
        if y.ndim == 1:
            y = y[:, None]

        # Define the kernel based on the specified kernel type
        if self.kernel == "Matern52":
            kernel = Matern52(input_dim=X.shape[-1], ARD=True)
        elif self.kernel == "Matern32":
            kernel = Matern32(input_dim=X.shape[-1], ARD=True)
        elif self.kernel == "RBF":
            kernel = RBF(input_dim=X.shape[-1], ARD=True)
        else:
            raise ValueError("Kernel should be 'RBF', 'Matern32', or 'Matern52'")

        # Create and optimize the GP regression model
        self.kmodel = GPRegression(X=X, Y=y, kernel=kernel, normalizer=self.normalizer)
        self.kmodel.optimize_restarts(num_restarts=self.num_restarts, messages=False)
        return self

    def predict(self, X, return_var: bool = False):
        """Predict using the Gaussian Process model.

        Args:
            X (ndarray): Input features of shape (n_samples, n_features).
            return_var (bool): Return the predictive variance.

        Returns:
            mean (ndarray): Predicted mean values for the input data
            or
            (mean, variance) (ndarray): Predicted mean and variance values if return_var is True.
        """
        # Get the mean prediction from the GP model
        mean, var = self.kmodel.predict(X)
        if return_var:
            return mean, np.tile(var, (1, mean.shape[-1]))
        else:
            return mean


def build_pipeline(apply_output_pca: bool = False) -> Pipeline:
    """Constructs a regression pipeline that includes:
    - PCA transformation on input features.
    - Standard scaling of input features.
    - Optional PCA transformation on the output.
    - Gaussian Process regression using `GPyRegressor`.

    Parameters:
    ----------
    apply_output_pca : bool, optional (default=False)
        If True, applies PCA on the output variables in addition to scaling.

    Returns:
    -------
    pipeline : Pipeline
        A scikit-learn pipeline with preprocessing and regression steps.
    """
    # define PCA transformation for input features
    pca_transformer = [
        (
            "pca",
            PCA(n_components=40),
            np.arange(2, 2 + nodes_train.shape[-1]),
        )
    ]

    input_preprocessor = ColumnTransformer(pca_transformer, remainder="passthrough")

    # define output preprocessing (scaling + optional PCA)
    output_preprocessor = Pipeline(
        steps=[
            ("scaler", MinMaxScaler()),
            ("pca", PCA(n_components=80))
            if apply_output_pca
            else ("identity", "passthrough"),
        ]
    )

    # define regressor with output transformation
    regressor = TransformedTargetRegressor(
        regressor=clone(GPyRegressor()),
        check_inverse=False,
        transformer=output_preprocessor,
    )

    # full pipeline with input preprocessing, scaling, and regression
    pipeline = Pipeline(
        steps=[
            ("preprocessor", input_preprocessor),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ]
    )

    return pipeline


if __name__ == "__main__":
    start = time.time()

    (
        nodes_train,
        X_scalars_train,
        Y_scalars_train,
        Y_density_train,
        Y_pressure_train,
        Y_temperature_train,
    ) = convert_data(ids_train)

    # Train
    X_train = np.concatenate([X_scalars_train, nodes_train], axis=-1)

    # scalars
    pipeline_scalars = build_pipeline(apply_output_pca=False)
    pipeline_scalars.fit(X_train, Y_scalars_train)
    # fields
    pipeline_density = build_pipeline(apply_output_pca=True)
    pipeline_density.fit(X_train, Y_density_train)

    pipeline_temperature = build_pipeline(apply_output_pca=True)
    pipeline_temperature.fit(X_train, Y_temperature_train)

    pipeline_pressure = build_pipeline(apply_output_pca=True)
    pipeline_pressure.fit(X_train, Y_pressure_train)

    print("duration train:", time.time() - start)
    start = time.time()

    # Predict

    (
        nodes_test,
        X_scalars_test,
        Y_scalars_test,
        Y_density_test,
        Y_pressure_test,
        Y_temperature_test,
    ) = convert_data(ids_test)

    X_test = np.concatenate([X_scalars_test, nodes_test], axis=-1)

    predictions = {}

    y_pred = pipeline_scalars.predict(X_test)
    predictions["Massflow"] = y_pred[:, 0]
    predictions["Compression_ratio"] = y_pred[:, 1]
    predictions["Efficiency"] = y_pred[:, 2]

    y_pred = pipeline_density.predict(X_test)
    predictions["Density"] = y_pred

    y_pred = pipeline_temperature.predict(X_test)
    predictions["Temperature"] = y_pred

    y_pred = pipeline_pressure.predict(X_test)
    predictions["Pressure"] = y_pred

    print("duration test:", time.time() - start)
    start = time.time()

    # dump
    reference = []
    for i, id in enumerate(ids_test):
        reference.append({})
        for fn in out_fields_names:
            reference[i][fn] = predictions[fn][i]
        for sn in out_scalars_names:
            reference[i][sn] = predictions[sn][i]

    with open("prediction.pkl", "wb") as file:
        pickle.dump(reference, file)

    # duration train: 416.71344685554504
    # duration test: 1.2284891605377197
