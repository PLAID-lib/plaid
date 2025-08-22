# processor.py

import numpy as np
from typing import List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from AM_POD import PolynomialManifoldApproximation

class InputProcessor:
    """
    Preprocesses input data by applying PCA to the mesh node coordinates
    and standard-scaling the resulting PCA components concatenated with
    additional input scalars.

    Args:
        explained_variance (float): The fraction of variance to preserve
            in the PCA decomposition (e.g. 0.99999).
    """
    def __init__(self, explained_variance: float):
        self.explained_variance = explained_variance
        self.pca: PCA = None
        self.scaler: StandardScaler = None
        self.n_components_: int = None
        self.node_shape_: tuple = None  # to reconstruct original shape

    def fit(self, inputs: dict[str, List[np.ndarray]]) -> "InputProcessor":
        """
        Fit the PCA on flattened node arrays and fit a StandardScaler
        on the concatenation of PCA components + the two input scalars.

        Args:
            inputs: dict with keys:
                - 'nodes': List of np.ndarray of shape (n_nodes, 2)
                - 'angle_in': List of floats
                - 'mach_out': List of floats

        Returns:
            self
        """
        # Extract raw data
        nodes_list = inputs['nodes']
        angle_list = inputs['angle_in']
        mach_list  = inputs['mach_out']

        # Remember original node shape for inverse transform
        self.node_shape_ = nodes_list[0].shape  # e.g. (36421, 2)

        # Flatten each nodes array to shape (n_nodes*2,)
        X_nodes = np.stack([n.reshape(-1) for n in nodes_list], axis=0)

        # PCA to reduce dimensionality, preserving desired variance
        self.pca = PCA(n_components=self.explained_variance, svd_solver='full')
        X_nodes_pca = self.pca.fit_transform(X_nodes)
        self.n_components_ = self.pca.n_components_

        # Build combined feature matrix: [ PCA components | angle_in | mach_out ]
        X_combined = np.hstack([
            X_nodes_pca,
            np.array(angle_list).reshape(-1, 1),
            np.array(mach_list).reshape(-1, 1)
        ])

        # Standard normalization
        self.scaler = StandardScaler()
        self.scaler.fit(X_combined)

        return self

    def transform(self, inputs: dict[str, List[np.ndarray]]) -> np.ndarray:
        """
        Apply the fitted PCA and StandardScaler to new data.

        Args:
            inputs: dict with same structure as in fit().

        Returns:
            A 2D numpy array of shape (n_samples, n_pca_components + 2).
        """
        nodes_list = inputs['nodes']
        angle_list = inputs['angle_in']
        mach_list  = inputs['mach_out']

        X_nodes = np.stack([n.reshape(-1) for n in nodes_list], axis=0)
        X_nodes_pca = self.pca.transform(X_nodes)

        X_combined = np.hstack([
            X_nodes_pca,
            np.array(angle_list).reshape(-1, 1),
            np.array(mach_list).reshape(-1, 1)
        ])

        return self.scaler.transform(X_combined)

    def fit_transform(self, inputs: dict[str, List[np.ndarray]]) -> np.ndarray:
        """
        Fit PCA and scaler on inputs, then transform and return processed data in one step.

        Args:
            inputs: dict with same structure as in fit().

        Returns:
            A 2D numpy array of shape (n_samples, n_pca_components + 2).
        """
        self.fit(inputs)
        return self.transform(inputs)

    def inverse_transform(self, X_transformed: np.ndarray) -> dict[str, List[Any]]:
        """
        Reconstruct approximate original nodes and scalars from the processed data.

        Args:
            X_transformed: Array of shape (n_samples, n_pca_components + 2)
                as output by transform().

        Returns:
            dict with keys:
                - 'nodes': List of np.ndarray of shape original (n_nodes, 2)
                - 'angle_in': List of floats
                - 'mach_out':  List of floats
        """
        # Undo standard scaling
        X_combined = self.scaler.inverse_transform(X_transformed)

        # Split back into PCA components and scalars
        X_nodes_pca = X_combined[:, :self.n_components_]
        angle_arr   = X_combined[:, self.n_components_]
        mach_arr    = X_combined[:, self.n_components_ + 1]

        # Inverse PCA to reconstruct flattened nodes
        X_nodes_flat = self.pca.inverse_transform(X_nodes_pca)

        # Reshape each to original mesh shape
        nodes_list = [
            flat.reshape(self.node_shape_)
            for flat in X_nodes_flat
        ]

        return {
            'nodes':     nodes_list,
            'angle_in':  list(angle_arr),
            'mach_out':  list(mach_arr)
        }


class OutputProcessor:
    """
    Preprocesses outputs by reducing mesh fields 'mach' and 'nut' via
    PolynomialManifoldApproximation and standard-scaling the concatenation
    of reduced fields and additional scalar outputs.

    Args:
        mach_params (tuple[int, int]): (polynomial_order, r) for 'mach' field.
        nut_params  (tuple[int, int]): (polynomial_order, r) for 'nut' field.
        podtype (str): 'pod', 'poly', or 'am' reduction type.
        reg_ls (float): L2 regularization for least squares.
        reg_nt (Optional[float]): L2 regularization for Newton solver.
        tol_rot (float): Tolerance for rotation convergence.
        tol_nt (float): Tolerance for Newton solver.
        max_iter_rot (int): Max iterations for rotation optimization.
        max_iter_nt (Optional[int]): Max iter for Newton loop.
        n_jobs (int): Number of parallel jobs.
        verbose (bool): Verbosity flag.
    """
    def __init__(
        self,
        mach_params: tuple[int, int],
        nut_params: tuple[int, int],
        podtype: str = 'am',
        reg_ls: float = 0.01,
        reg_nt: Optional[float] = None,
        tol_rot: float = 1e-6,
        tol_nt: float = 1e-8,
        max_iter_rot: int = 100,
        max_iter_nt: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = False
    ):
        polynomial_order_m, r_m = mach_params
        polynomial_order_n, r_n = nut_params
        self.pma_mach = PolynomialManifoldApproximation(
            polynomial_order=polynomial_order_m,
            r=r_m,
            q=None,
            podtype=podtype,
            reg_ls=reg_ls,
            reg_nt=reg_nt,
            tol_rot=tol_rot,
            tol_nt=tol_nt,
            max_iter_rot=max_iter_rot,
            max_iter_nt=max_iter_nt,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.pma_nut = PolynomialManifoldApproximation(
            polynomial_order=polynomial_order_n,
            r=r_n,
            q=None,
            podtype=podtype,
            reg_ls=reg_ls,
            reg_nt=reg_nt,
            tol_rot=tol_rot,
            tol_nt=tol_nt,
            max_iter_rot=max_iter_rot,
            max_iter_nt=max_iter_nt,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.scaler: StandardScaler = None
        self.r_mach = r_m
        self.r_nut = r_n

    def fit(
        self,
        outputs_train: dict[str, List[Any]],
        outputs_test: dict[str, List[Any]]| None = None
    ) -> "OutputProcessor":
        """
        Fit PMA models on train/test data and a StandardScaler on combined features.
        """
        # Prepare field snapshots (samples x features)
        S_train_mach = np.stack(outputs_train['mach'], axis=0)
        S_train_nut  = np.stack(outputs_train['nut'], axis=0)
        if outputs_test is not None:
            S_test_mach  = np.stack(outputs_test['mach'], axis=0)
            S_test_nut   = np.stack(outputs_test['nut'], axis=0)
        else:
            S_test_mach = None
            S_test_nut = None

        # Fit manifold approximations
        self.pma_mach.fit(S_train_mach, S_test_mach)
        self.pma_nut.fit(S_train_nut,  S_test_nut)

        # Transform fields
        X_mach = self.pma_mach.transform(S_train_mach)
        X_nut  = self.pma_nut.transform(S_train_nut)

        # Stack scalar outputs
        scalar_keys = ['Q', 'power', 'Pr', 'Tr', 'eth_is', 'angle_out']
        X_scalars = np.vstack([np.array(outputs_train[k]) for k in scalar_keys]).T

        # Combine and scale
        X = np.hstack([X_mach, X_nut, X_scalars])
        self.scaler = StandardScaler().fit(X)
        return self

    def transform(
        self,
        outputs: dict[str, List[Any]]
    ) -> np.ndarray:
        """
        Transform outputs using fitted PMA and scaler.
        """
        S_mach = np.stack(outputs['mach'], axis=0)
        S_nut  = np.stack(outputs['nut'],  axis=0)
        X_mach = self.pma_mach.transform(S_mach)
        X_nut  = self.pma_nut.transform(S_nut)
        scalar_keys = ['Q', 'power', 'Pr', 'Tr', 'eth_is', 'angle_out']
        X_scalars = np.vstack([np.array(outputs[k]) for k in scalar_keys]).T
        X = np.hstack([X_mach, X_nut, X_scalars])
        return self.scaler.transform(X)

    def fit_transform(
        self,
        outputs_train: dict[str, List[Any]],
        outputs_test: dict[str, List[Any]] | None = None
    ) -> np.ndarray:
        """
        Fit PMA and scaler then transform train outputs.
        """
        self.fit(outputs_train, outputs_test)
        return self.transform(outputs_train)

    def inverse_transform(
        self,
        X_transformed: np.ndarray
    ) -> dict[str, List[Any]]:
        """
        Inverse transform to reconstruct approximate original outputs.
        """
        X_comb = self.scaler.inverse_transform(X_transformed)
        n1 = self.r_mach
        n2 = self.r_nut
        X_mach = X_comb[:, :n1]
        X_nut  = X_comb[:, n1:n1+n2]
        scalars = X_comb[:, n1+n2:]

        mach_arr = self.pma_mach.inverse_transform(X_mach)
        nut_arr  = self.pma_nut.inverse_transform(X_nut)
        mach_list = [mach_arr[i] for i in range(mach_arr.shape[0])]
        nut_list  = [nut_arr[i]  for i in range(nut_arr.shape[0])]
        scalar_keys = ['Q', 'power', 'Pr', 'Tr', 'eth_is', 'angle_out']
        scalar_dict = {k: list(scalars[:, idx]) for idx, k in enumerate(scalar_keys)}

        return {
            'mach': mach_list,
            'nut': nut_list,
            **scalar_dict
        }
