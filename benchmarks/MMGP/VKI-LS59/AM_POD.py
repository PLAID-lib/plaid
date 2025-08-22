import time
from itertools import combinations_with_replacement

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares


class PolynomialManifoldApproximation:
    def __init__(
        self,
        polynomial_order,
        r,
        q=None,
        podtype="am",
        reg_ls=1e-3,
        reg_nt=None,
        tol_rot=1e-6,
        tol_nt=1e-8,
        max_iter_rot=100,
        max_iter_nt=None,
        n_jobs=-1,
        verbose=False,
    ):
        """This is an implementation of the paper https://arxiv.org/pdf/2306.13748.

        Initialize the class for polynomial manifold approximation.

        Parameters:
            polynomial_order (int): Maximum order of the polynomial (P).
            r (int): Number of principal modes to retain in the POD base.
            q (int or None): number of complements modes to retain in the POD base.
            podtype (str):  - pod: for pod reduction,
                            - poly: pod + polynomial correction reduction,
                            - am: for adaptative manifold reduction
            reg_ls (float): Regularization parameter for the coefficients (L2 penalty) in least square problem.
            reg_nt (float or None): Regularization parameter for the coefficients (L2 penalty) in Newthon solver.
            tol_rot (float): Tolerance for iterative convergence for the rotation space.
            tol_nt (float): Tolerance for iterative convergence for the Newton.
            max_iter_rot (int): Maximum number of iterations for the optimization loop of the rotation matrix.
            max_iter_nt (int): Maximum number of iterations for the Newthon loop.
            n_jobs (int): number of cpu
            verbose (bool): True to display optimization iterations.
        """
        assert podtype in ["pod", "poly", "am"]
        if polynomial_order == 1 and podtype != "pod":
            if verbose:
                print(
                    f"WARNING: podtype is set to <{podtype:s}> but polynomial_order is <{polynomial_order:1}>. A podtype <pod> will be used instead"
                )
            podtype = "pod"
        self.polynomial_order = polynomial_order
        self.r = r
        self.q = q if q is None else r + q
        self.type = podtype
        self.reg_ls = reg_ls
        self.reg_nt = reg_nt
        self.tol_rot = tol_rot
        self.tol_nt = tol_nt if tol_nt > 0 or tol_nt is None else 1e-8
        self.max_iter_rot = max_iter_rot
        self.max_iter_nt = max_iter_nt
        self.n_jobs = n_jobs
        self.mean = None
        self.V = None  # Principal modes (from SVD)
        self.Vb = None  # Extra principal modes (from SVD)
        self.psi = None  # polynomial correction
        self.is_fitted = False
        self.is_start = False
        self.verbose = verbose

    def _center_data(self, S):
        """Center the data by subtracting the mean from each feature.

        Parameters:
            S (np.ndarray): Snapshot matrix (N, Ns).

        Returns:
            np.ndarray: Centered snapshot matrix.
        """
        if not self.is_fitted:
            self.mean = np.mean(S, axis=1, keepdims=True)
        return S - np.tile(self.mean, (1, S.shape[1]))

    def reduce(self, S):
        """POD projection.

        Parameters:
            S (np.ndarray): Snapshot matrix (N, Ns).

        Returns:
            np.ndarray: reduced Snapshot matrix (r, Ns).
        """
        return self.V.T @ S

    def _generate_combinations(self, coefficients):
        """Generate the W matrix containing all polynomial combinations up to the specified order.

        Parameters:
            coefficients (np.ndarray): Coefficients from the projection (r, Ns).

        Returns:
            W (np.ndarray): Array of polynomial combinations. (p, Ns)
        """
        coeff_dim = coefficients.shape[0]
        num_snapshots = coefficients.shape[1]

        # Genera tutte le combinazioni di indici fino al grado polinomiale specificato
        combs = list(
            combinations_with_replacement(range(coeff_dim), 2)
        )  # Polynomial degree 2

        for deg in range(3, self.polynomial_order + 1):  # Extend for higher degrees
            combs.extend(combinations_with_replacement(range(coeff_dim), deg))

        num_combinations = len(combs)
        W = np.empty((num_combinations, num_snapshots))

        # Funzione per calcolare la colonna idx di W
        def compute_column(snapshot):
            return np.array([np.prod(snapshot[list(combo)]) for combo in combs])

        if num_snapshots > 1:
            # Parallelizzazione solo se ci sono più di una colonna
            results = Parallel(n_jobs=-1)(
                delayed(compute_column)(snapshot) for snapshot in coefficients.T
            )
            W[:, :] = np.column_stack(results)
        else:
            # Se c'è solo una colonna, esegui il calcolo sequenziale
            W[:, 0] = compute_column(coefficients.T[0])

        return W

    def score(self, S_test):
        S_test = S_test.T
        assert self.is_fitted
        norm_S_test = np.linalg.norm(self._center_data(S_test), ord="fro")
        S_reduced_test = self.reduce(self._center_data(S_test))
        W_test = self._generate_combinations(S_reduced_test)
        return abs(
            1
            - np.linalg.norm(
                self.V @ S_reduced_test + self.Vb @ self.psi @ W_test, ord="fro"
            )
            / norm_S_test
        )

    def fit(self, S, S_test=None):
        """Fit the model by iteratively solving the optimization loop.

        Parameters:
            S (np.ndarray): Snapshot matrix (N, Ns).
        """
        S = S.T
        if S_test is not None:
            S_test = S_test.T
        assert self.r < S.shape[1]
        if self.q is not None:
            assert self.q < S.shape[1]

        S_centered = self._center_data(S)  # (N x Ns)
        norm_S = np.linalg.norm(S_centered, ord="fro")
        if S_test is not None:
            norm_S_test = np.linalg.norm(self._center_data(S_test), ord="fro")
        # Initial SVD and initialization
        U, sig, _ = np.linalg.svd(S_centered, full_matrices=False)
        if self.verbose:
            print(40 * "=")
        if self.verbose:
            print(
                f"POD computed cumulated expected variance : {np.cumsum(sig**2)[self.r] / np.sum(sig**2):.4f}"
            )
        if self.verbose:
            print(40 * "=")
        if not self.is_start:
            # compute standard POD
            self.V = U[:, : self.r]  # (N x r)
            self.Vb = U[:, self.r : self.q]  # (N x q)
            S_reduce = self.reduce(S_centered)  # (r x Ns)
            error_pod = 1 - np.linalg.norm(self.V @ S_reduce, ord="fro") / norm_S

            # compute polynomial correction
            now = time.time()
            W = self._generate_combinations(S_reduce)
            self.psi = np.linalg.lstsq(
                W @ W.T + self.reg_ls * np.eye(W.shape[0]),
                (self.Vb.T @ S_centered @ W.T).T,
                rcond=None,
            )[0].T  # (q x Ns)
            polynomial_error = (
                1
                - np.linalg.norm(self.V @ S_reduce + self.Vb @ self.psi @ W, ord="fro")
                / norm_S
            )
            SW = np.concatenate([S_reduce.T, (self.psi @ W).T], axis=1)
            old_error = np.copy(polynomial_error)
            if self.verbose:
                print(
                    f"---> compute polynomial correction {(time.time() - now):.4f} seconds"
                )
            if self.type == "pod":
                self.psi *= 0
                self.is_fitted = True
                return self
            elif self.type == "poly":
                self.is_fitted = True
                return self
            self.is_start = True
            self.iter_step = 0
        else:
            # compute standard POD
            V = U[:, : self.r]  # (N x r)
            Vb = U[:, self.r : self.q]  # (N x q)
            S_reduce = self.reduce(S_centered)  # (r x Ns)
            error_pod = 1 - np.linalg.norm(V @ S_reduce, ord="fro") / norm_S
            # compute polynomial correction
            now = time.time()
            W = self._generate_combinations(S_reduce)
            psi = np.linalg.lstsq(
                W @ W.T + self.reg_ls * np.eye(W.shape[0]),
                (Vb.T @ S_centered @ W.T).T,
                rcond=None,
            )[0].T  # (q x Ns)
            polynomial_error = (
                1 - np.linalg.norm(V @ S_reduce + Vb @ psi @ W, ord="fro") / norm_S
            )
            if self.verbose:
                print(
                    f"---> compute polynomial correction {(time.time() - now):.4f} seconds"
                )
            # restore last AM iteration
            now = time.time()
            S_reduce = self.restart_reduce
            W = self._generate_combinations(S_reduce)
            SW = np.concatenate([S_reduce.T, (self.psi @ W).T], axis=1)
            old_error = (
                1
                - np.linalg.norm(self.V @ S_reduce + self.Vb @ self.psi @ W, ord="fro")
                / norm_S
            )
            if self.verbose:
                print(
                    f"---> restore last AM iteration {(time.time() - now):.4f} seconds"
                )

        if S_test is not None:
            S_reduced_test = self.reduce(self._center_data(S_test))
            W_test = self._generate_combinations(S_reduced_test)
            old_error_test = abs(
                1
                - np.linalg.norm(
                    self.V @ S_reduced_test + self.Vb @ self.psi @ W_test, ord="fro"
                )
                / norm_S_test
            )
            if self.verbose:
                print(
                    f"---> pod {error_pod:.4e} vs polynomial {polynomial_error:.4e} vs AM-polynomial {old_error:.4e} vs test {old_error_test:.4e}"
                )
        else:
            if self.verbose:
                print(
                    f"---> pod {error_pod:.4e} vs polynomial {polynomial_error:.4e} vs AM-polynomial {old_error:.4e}"
                )

        if self.verbose:
            print(40 * "=")
        if self.verbose:
            print("initialization completed")
        if self.verbose:
            print(40 * "=")

        for iter_loop in range(self.max_iter_rot):
            if self.verbose:
                print(f"AM iteration {self.iter_step}")
            V_old = np.copy(self.V)
            Vb_old = np.copy(self.Vb)
            psi_old = np.copy(self.psi)

            # Step 1: Solve the Orthogonal Procrustes Problem
            now = time.time()
            Omega = self.solve_orthogonal_procrustes(S_centered, SW)  # (N x [r+q])
            if self.verbose:
                print(
                    f"---> solve procrustes problem in {(time.time() - now):.4f} seconds"
                )

            # Step 2: Update V and Vb
            self.V = Omega[:, : self.r]
            self.Vb = Omega[:, self.r :]

            # Step 3: Optimize SW using Levenberg-Marquardt
            now = time.time()
            S_reduce = self._levenberg_marquardt(S_centered, Omega)
            self.restart_reduce = S_reduce
            if self.verbose:
                print(
                    f"---> solve levenberg marquardt problem in {(time.time() - now):.4f} seconds"
                )

            # Step 4: Compute W, and psi
            W = self._generate_combinations(S_reduce)  # (p x Ns)

            now = time.time()
            self.psi = np.linalg.lstsq(
                W @ W.T + self.reg_ls * np.eye(W.shape[0]),
                (self.Vb.T @ S_centered @ W.T).T,
                rcond=None,
            )[0].T  # (q x Ns)
            if self.verbose:
                print(
                    f"---> least square problem solved in {(time.time() - now):.4f} seconds"
                )

            # Step 5: Compute SW and approximation error
            SW = np.concatenate([S_reduce.T, (self.psi @ W).T], axis=1)  # (Ns x [r+q])
            new_error = (
                1
                - np.linalg.norm(self.V @ S_reduce + self.Vb @ self.psi @ W, ord="fro")
                / norm_S
            )
            if S_test is None:
                if self.verbose:
                    print(
                        f"---> pod {error_pod:.4e} vs polynomial {polynomial_error:.4e} vs AM-polynomial {new_error:.4e}"
                    )
            else:
                S_reduced_test = self.reduce(self._center_data(S_test))
                W_test = self._generate_combinations(S_reduced_test)
                new_error_test = abs(
                    1
                    - np.linalg.norm(
                        self.V @ S_reduced_test + self.Vb @ self.psi @ W_test, ord="fro"
                    )
                    / norm_S_test
                )
                if self.verbose:
                    print(
                        f"---> pod {error_pod:.4e} vs polynomial {polynomial_error:.4e} vs AM-polynomial {new_error:.4e} vs test {new_error_test:.4e}"
                    )

            if S_test is not None and (new_error_test > old_error_test):
                if self.verbose:
                    print("early stop criteria")
                self.V = np.copy(V_old)
                self.Vb = np.copy(Vb_old)
                self.psi = np.copy(psi_old)
                self.is_fitted = True
                break
            elif abs(old_error - new_error) < self.tol_rot:
                if self.verbose:
                    print("converged criteria")
                self.is_fitted = True
                break

            old_error = np.copy(new_error)
            if S_test is not None:
                old_error_test = np.copy(new_error_test)
            self.iter_step += 1

        self.is_fitted = True
        return self

    def solve_orthogonal_procrustes(self, S_centered, SW):
        """Solve the Orthogonal Procrustes Problem: minimize ||S_centered - Omega SW||
        subject to Omega^T Omega = I.

        Parameters:
            S_centered (np.ndarray): Centered snapshot matrix (N, Ns).
            SW (np.ndarray): concatenated [S_reduced.T (psi@W).T] (Ns, Ns)

        Returns:
            np.ndarray: Orthogonal matrix Omega of shape (N, Ns).
        """
        A = np.dot(S_centered, SW)
        U, _, Vt = np.linalg.svd(A, full_matrices=False)
        Omega = np.dot(U, Vt)

        return Omega

    def _levenberg_marquardt(self, S, Omega):
        """Solve the minimization problem ||S - Omega X(coefficients)|| using Levenberg-Marquardt,
        parallelizing the computation over columns of S.

        Parameters:
            S (np.ndarray): Snapshot vector of shape (N, Ns).
            Omega (np.ndarray): Matrix of shape (N, r + p) such that Omega^T Omega = I.

        Returns:
            S_reduced (np.ndarray): Optimized coefficients vector of shape (r, Ns).
        """
        num_cols = S.shape[1]  # Number of columns in S
        S_reduced = self.reduce(S)  # Reduce the dimension of S

        def optimize_column(i):
            """Optimize a single column using Levenberg-Marquardt."""
            x0 = S_reduced[:, i].ravel()  # Flatten initial guess
            result = least_squares(
                lambda x: self._residu(x, S[:, i], Omega),
                x0,
                jac=lambda x: self._jacobian(x, S[:, i], Omega),
                method="trf",
                max_nfev=self.max_iter_nt * S.shape[0] * self.polynomial_order
                if self.max_iter_nt is not None
                else None,
                ftol=self.tol_nt,
                xtol=self.tol_nt,
                gtol=self.tol_nt,
            )
            return result.x  # Return optimized solution for this column

        # Run parallel optimization for each column
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(optimize_column)(i) for i in range(num_cols)
        )

        # Convert list of results into an array
        X_opt = np.column_stack(results)

        return X_opt  # Return the optimized matrix

    def _residu(self, coefficients, S_col, Omega):
        """Compute the residual for a single column of S in the Levenberg-Marquardt optimization.

        Parameters:
            coefficients (np.ndarray): Parameters to be optimized (in the reduced space).
            S_col (np.ndarray): A single column of the original data matrix S.
            Omega (np.ndarray): Weight matrix.

        Returns:
            np.ndarray: Residuals as a 1D array.
        """
        coefficients = coefficients.reshape(-1, 1)
        g = self._generate_combinations(
            coefficients
        )  # Generate polynomial combinations
        X = np.concatenate([coefficients, self.psi @ g], axis=0)  # Construct the matrix

        resid = (
            Omega.T @ S_col.reshape(-1, 1) - X
        )  # Compute the residual for this column
        return resid.ravel()  # Return a 1D vector for least_squares

    def _jacobian(self, coefficients, S_col, Omega):
        """Compute the Jacobian matrix of the residual function.

        Parameters:
            coefficients (np.ndarray): Parameters to be optimized (in the reduced space).
            S_col (np.ndarray): A single column of the original data matrix S.
            Omega (np.ndarray): Weight matrix.

        Returns:
            np.ndarray: Jacobian matrix of shape (num_residuals, num_parameters).
        """
        coefficients = coefficients.reshape(-1, 1)  # Ensure correct shape

        # Compute Jacobian with respect to coefficients
        J_g = self._compute_combinations_jacobian(
            coefficients
        )  # Custom function to compute derivatives

        J_psi_g = self.psi @ J_g  # Apply transformation psi

        # Construct the full Jacobian matrix
        J = -np.concatenate([np.eye(coefficients.shape[0]), J_psi_g], axis=0)

        if self.reg_nt is None or self.reg_nt == 0.0:
            return J
        else:
            JtJ = J.T @ J  # Shape (r, r)
            diag_JtJ = np.diag(JtJ)  # Extract diagonal

            # Add regularization term
            regularized_JtJ = JtJ + self.reg_nt * np.diag(
                diag_JtJ
            )  # Regularized J.T @ J
            # Compute Cholesky decomposition
            L = np.linalg.cholesky(regularized_JtJ).T  # Shape (r, r)

            # Solve for a modified J with the same shape as the original J (N, r)
            J = J @ np.linalg.inv(L)
            return J

    def _compute_combinations_jacobian(self, coefficients):
        """Compute the Jacobian of the polynomial combinations matrix W with respect to coefficients.

        Parameters:
            coefficients (np.ndarray): Coefficients from the projection (r, Ns).

        Returns:
            J (np.ndarray): Jacobian matrix of polynomial combinations. Shape: (p, r, Ns)
                            where p is the number of polynomial combinations.
        """
        coeff_dim = coefficients.shape[0]  # Number of coefficients (r)
        combs = []  # store combinations

        # Generate all polynomial combinations
        for deg in range(2, self.polynomial_order + 1):
            combs.extend(combinations_with_replacement(range(coeff_dim), deg))

        num_combinations = len(combs)  # Total number of polynomial terms
        J = np.zeros(
            (num_combinations, coeff_dim, coefficients.shape[1])
        )  # Jacobian storage

        # Compute derivatives for each column (snapshot)
        for idx, snapshot in enumerate(coefficients.T):  # Iterate over snapshots
            for row_idx, combo in enumerate(
                combs
            ):  # Iterate over polynomial combinations
                combo = list(combo)  # Convert to list for indexing

                # Compute the polynomial term itself
                w_i = np.prod(snapshot[combo])

                # Compute derivatives w.r.t. each coefficient
                for j in range(coeff_dim):  # Iterate over coefficients
                    if j in combo:
                        # Compute derivative using the product rule
                        partial_derivative = (
                            w_i / snapshot[j] * combo.count(j)
                            if snapshot[j] != 0
                            else 0
                        )
                        J[row_idx, j, idx] = partial_derivative  # Store in Jacobian

        return J.squeeze()  # Shape: (p, r, Ns)

    def transform(self, S):
        S = S.T
        assert self.is_fitted
        assert self.V.shape[0] == S.shape[0]

        S_centered = self._center_data(S)
        S_reduce = self.reduce(S_centered)
        return S_reduce.T

    def fit_transform(self, S_train, S_test):
        self = self.fit(S_train, S_test)
        return self.transform(S_train)

    def inverse_transform(self, S_reduce):
        S_reduce = S_reduce.T
        assert self.is_fitted
        assert self.V.shape[1] == S_reduce.shape[0]
        W = self._generate_combinations(S_reduce)
        S = (
            self.V @ S_reduce
            + self.Vb @ self.psi @ W
            + np.tile(self.mean, (1, S_reduce.shape[1]))
        )
        return S.T
