import GPy
import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, RegressorMixin, clone


class GPyRegressor(BaseEstimator, RegressorMixin):
    """
    Custom Gaussian Process Regressor using GPy library.

    Args:
        normalizer (bool): Whether to normalize the output.
        constant_mean (bool): Whether to use a constant mean model.
        kernel (str): Type of kernel to use in the Gaussian Process.
            Options: 'Matern52', 'Matern32', 'Rbf'.
        num_restarts (int): Number of restarts for kernel optimization.
    """

    def __init__(
        self,
        normalizer: bool = False,
        constant_mean: bool = False,
        kernel: str = "Matern52",
        num_restarts: int = 5,
    ):
        self.normalizer = normalizer
        self.constant_mean = constant_mean
        self.kernel = kernel
        self.num_restarts = num_restarts

    def fit(self, X, y):
        """
        Fit the Gaussian Process model to the data.

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
            kernel = GPy.kern.Matern52(input_dim=X.shape[-1], ARD=True)
        elif self.kernel == "Matern32":
            kernel = GPy.kern.Matern32(input_dim=X.shape[-1], ARD=True)
        elif self.kernel == "RBF":
            kernel = GPy.kern.RBF(input_dim=X.shape[-1], ARD=True)
        else:
            raise ValueError("Kernel should be 'RBF', 'Matern32', or 'Matern52'")

        mean_function = None
        if self.constant_mean:
            mean_function = GPy.mappings.Constant(
                input_dim=X.shape[-1], output_dim=y.shape[-1]
            )

        # Create and optimize the GP regression model
        self.kmodel = GPy.models.GPRegression(
            X=X,
            Y=y,
            kernel=kernel,
            normalizer=self.normalizer,
            mean_function=mean_function,
        )
        self.kmodel.optimize_restarts(num_restarts=self.num_restarts, messages=False)
        return self

    def predict(self, X, return_var: bool = False):
        """
        Predict using the Gaussian Process model.

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
