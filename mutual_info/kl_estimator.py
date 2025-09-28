# Mutual information estimation with Kozachenko-Leonenko (KL) estimator.
# The following code is based on the ICLR 2024 paper:
# "Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression".
# https://openreview.net/forum?id=huGECz8dPp

import math
from collections import Counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm
from scipy.special import loggamma
from sklearn.neighbors import BallTree
from torch import Tensor


class Functional:
    """Class for evaluating functional based on density estimates.

    Args:
        atol: float
            Absolute tolerance.
        rtol: float
            Relative tolerance.

    """

    def __init__(self, atol: float = 0.0, rtol: float = 0.0) -> None:
        self.atol = atol
        self.rtol = rtol

    def fit(self, x: NDArray | Tensor) -> None:
        """Build a density estimate from the data.

        Args:
            x : array_like
                I.i.d. samples.

        """
        self.data = x

    def get_loo_densities(self) -> NDArray:
        """Obtain the density at the points on which the fitting was performed.

        The leave-one-out method is applied.
        """
        raise NotImplementedError

    def integrate(
        self,
        func: callable,
        bootstrap_size: int | None = None,
    ) -> tuple[float, float]:
        """Functional evaluation according to the leave-one-out method.

        Args:
            func: callable
                Integrated function.
            bootstrap_size: int
                Bootstrap sample size.

        """
        n_samples, dim = self.tree_.data.shape

        # Obtain density values.
        densities = self.get_loo_densities()
        if densities is None:
            return np.nan, np.nan

        if bootstrap_size is None:
            # Functional evaluation by simple averaging.
            values = self._get_values(func, densities)

            # The mean and variance of the functional.
            mean = math.fsum(values) / n_samples
            std = np.std(values) / np.sqrt(n_samples)

        else:
            # Functional evaluation using the bootstrap method.
            values = []
            rng = np.random.default_rng()
            for _ in range(bootstrap_size):
                values.append(
                    math.fsum(
                        self._get_values(func, rng.choice(densities, size=n_samples))
                        / n_samples,
                    ),
                )

            # The mean and variance of the functional.
            mean = np.mean(values)
            std = np.std(values)

        return mean, std

    def _get_values(self, func: callable, densities: NDArray) -> NDArray:
        """Calculate function values.

        Args:
            func: callable
                Integrated function.
            densities: array_like
                Density function values at corresponding points.

        Returns:
            NDArray (n_samples,)
                Function values at the points on which the fitting was performed.

        """
        # If the density array is one-dimensional, add a dummy axis
        # - generalization to the weighted case.
        if len(densities.shape) == 1:
            densities = densities[:, np.newaxis]

        # Weights.
        n_components = densities.shape[1]
        if not hasattr(self, "weights"):
            weights = np.zeros(n_components)
            weights[0] = 1.0
        else:
            weights = self.weights

        # Evaluation.
        return func(densities) @ weights


class KLFunctional(Functional):
    """Class for evaluating functional based on Kozachenko-Leonenko (KL) estimator."""

    def __init__(
        self,
        *args: float,
        k_neighbors: int = 5,
        tree_algorithm: str = "ball_tree",
        tree_params: dict | None = None,
    ) -> None:
        """Initialize the class.

        Args:
            *args : float
                Additional positional arguments passed to the base Functional class.
            k_neighbors : int
                The number of nearest neighbors used to estimate the density.
            tree_algorithm : str
                Metric tree used for density estimation.
            tree_params : dict
                Metric tree parameters.

        """
        if tree_params is None:
            tree_params = {"leaf_size": 40, "metric": "euclidean"}
        if k_neighbors <= 0:
            msg = "Number of neighbors must be positive"
            raise ValueError(msg)

        super().__init__(*args)

        self.k_neighbors = k_neighbors
        self.tree_algorithm = tree_algorithm
        self.tree_params = tree_params

        self.weights = np.zeros(self.k_neighbors)
        self.weights[0] = 1.0

    def fit(
        self,
        x: NDArray[Any] | Tensor,
        *,
        fit_weights: bool = True,
    ) -> None:
        """Build a kNN density estimate.

        Args:
            x : array_like
                I.i.d. samples.
            fit_weights : bool
                Do the weights selection.

        """
        if len(x.shape) != 2 or x.shape[0] < self.k_neighbors:  # noqa: PLR2004
            # print(x.shape, self.k_neighbors)
            msg = (
                "Input x must be of shape (n_samples, n_features) and "
                "n_samples >= k_neighbors"
            )
            raise ValueError(msg)

        self.data = x

        if self.tree_algorithm == "ball_tree":
            self.tree_ = BallTree(
                x.cpu() if isinstance(x, Tensor) else x, **self.tree_params
            )
        else:
            raise NotImplementedError

        # Select the weights.
        if fit_weights:
            self.set_optimal_weights()

    def get_loo_densities(self) -> NDArray:
        """Obtain the density at the points on which the fitting was performed.

        The leave-one-out method is applied.

        Returns:
            NDArray (n_samples, k_neighbors)
                Density values at the points on which the fitting was performed.

        """
        n_samples, dim = self.tree_.data.shape
        # print(n_samples, dim, self.k_neighbors)
        # Getting `_k_neighbors` nearest neighbors.
        distances, indexes = self.tree_.query(
            self.tree_.data,
            self.k_neighbors + 1,
            return_distance=True,
        )
        # Remove the first column of distances, which corresponds to the point itself.
        distances = distances[:, 1:]

        # Calculate the volume of the unit ball in the given dimension.
        unit_ball_volume = (np.pi ** (0.5 * dim)) / math.gamma(0.5 * dim + 1.0)

        # Calculate the digamma function values.
        psi = np.zeros(self.k_neighbors)
        psi[0] = -np.euler_gamma
        for index in range(1, self.k_neighbors):
            psi[index] = psi[index - 1] + 1 / index

        densities = np.exp(psi) / (
            unit_ball_volume
            * np.clip(np.power(distances, dim), a_min=1e-12, a_max=None)
        )

        # Normalization.
        densities /= n_samples - 1

        return densities

    def set_optimal_weights(
        self,
        rcond: float = 1e-6,
        *,
        zero_constraints: bool = True,
    ) -> NDArray:
        """Optimal weights selection.

        Args:
            rcond: float
                Cut-off ratio for small singular values in least squares method.
            zero_constraints: bool
                Add constraints, zeroing some of the weights.

        """
        n_samples, dim = self.tree_.data.shape

        if dim <= 4:  # noqa: PLR2004
            # If the number of utilized neighbors is small, the weights are trivial.
            self.weights = np.zeros(self.k_neighbors)
            self.weights[0] = 1.0

        else:
            # Build a linear constraint.
            constraints = []

            # Constraint: the sum equals one.
            constraints.append(np.ones(self.k_neighbors) / self.k_neighbors)

            # Constraint: gamma function.
            n_gamma_constraints = dim // 4
            for k in range(1, n_gamma_constraints + 1):
                constraints.append(
                    np.exp(
                        loggamma(np.arange(1, self.k_neighbors + 1) + 2 * k / dim)
                        - loggamma(np.arange(1, self.k_neighbors + 1)),
                    ),
                )
                constraints[-1] /= np.linalg.norm(constraints[-1])

            # Constraint: zero out some elements.
            if zero_constraints:
                nonzero = {i * self.k_neighbors // dim - 1 for i in range(1, dim + 1)}
                for j in range(self.k_neighbors):
                    if j not in nonzero:
                        constraint = np.zeros(self.k_neighbors)
                        constraint[j] = 1.0
                        constraints.append(constraint)

            constraints = np.vstack(constraints)

            # Right hand side.
            rhs = np.zeros(constraints.shape[0])
            rhs[0] = 1.0 / self.k_neighbors

            self.weights = np.linalg.lstsq(constraints, rhs, rcond=rcond)[0]

        return self.weights


class EntropyEstimator:
    """Class for entropy estimation.

    Currently, only the Kozachenko-Leonenko (KL) estimator is supported.
    """

    def __init__(
        self,
        *,
        rescale: bool = False,
        method: str = "KL",
        functional_params: dict | None = None,
    ) -> None:
        """Initialize the estimator.

        Args:
            rescale: Enables data normalization step.
            method: Entropy estimation method to use (currently only "KL" is supported).
            functional_params: Additional parameters passed to the functional evaluator.

        """
        self.rescale = rescale
        self._scaling_matrix = None
        self._scaling_delta_entropy = 0.0
        self._fitted = False

        if method != "KL":
            msg = f"Entropy estimation method {method} is not supported."
            raise NotImplementedError(msg)
        self._functional = KLFunctional(**functional_params)

    def fit(
        self,
        data: NDArray | Tensor,
        *,
        fit_scaling_matrix: bool = True,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Fit the estimator to the data.

        Args:
            data: I.i.d. samples from the random variable.
            fit_scaling_matrix : bool, optional
                Fit matrix for data normalization.
            **kwargs: Additional keyword arguments passed to the functional evaluator.

        """
        # Data normalization.
        if self.rescale:
            if fit_scaling_matrix:
                # Covariance matrix (required for normalization).
                # It is taken into account that in the case of one-dimensional data,
                # np.cov returns a number.
                cov_matrix = np.cov(data, rowvar=False)
                if data.shape[1] == 1:
                    cov_matrix = np.array([[cov_matrix]])

                # Getting the scaling matrix from the covariance matrix.
                self._scaling_matrix = np.linalg.inv(sqrtm(cov_matrix))
                self._scaling_delta_entropy = -0.5 * np.linalg.slogdet(cov_matrix)[1]
            data = data @ self._scaling_matrix

        # Functional evaluator.
        self._functional.fit(data, **kwargs)
        self._fitted = True

    def estimate(
        self, data: NDArray | Tensor | None = None
    ) -> tuple[float, float]:  # noqa: ARG002
        # print("estimating")
        """Entropy estimation."""
        if not self._fitted:
            msg = "Entropy estimator is not fitted"
            raise RuntimeError(msg)
        # The evaluation itself is performed by the functional evaluator.
        mean, std = self._functional.integrate(np.log)
        # print("functional done", mean, std)

        # Recall that h(AX) = h(X) + log|det(A)|, where A is the scaling matrix.
        # Thus, the original h(X) is obtained by subtracting the log-determinant term.
        return -mean - self._scaling_delta_entropy, std


class MIEstimator:
    """Mutual information estimator."""

    def __init__(
        self,
        *,
        x_is_discrete: bool = False,
        y_is_discrete: bool = False,
        entropy_estimator_params: dict | None = None,
    ) -> None:
        if entropy_estimator_params is None:
            entropy_estimator_params = {"method": "KL", "functional_params": None}
        self._x_is_discrete = x_is_discrete
        self._y_is_discrete = y_is_discrete
        self.entropy_estimator_params = entropy_estimator_params

        self._x_entropy_estimator = None
        self._y_entropy_estimator = None
        self._x_y_entropy_estimator = None

    def fit(self, x: NDArray | Tensor, y: NDArray | Tensor) -> None:
        """Fit parameters of the estimator.

        Args:
            x: I.i.d. samples from X.
            y: I.i.d. samples from Y.

        """
        if x.shape[0] != y.shape[0]:
            msg = "X and Y must have the same length"
            raise ValueError(msg)

        if not self._x_is_discrete and not self._y_is_discrete:
            if self.entropy_estimator_params["method"] == "KL":
                self._x_y_entropy_estimator = EntropyEstimator(
                    **self.entropy_estimator_params,
                )
                self._x_y_entropy_estimator.fit(
                    np.concatenate([x.cpu(), y.cpu()], axis=1)
                )

                self._x_entropy_estimator = EntropyEstimator(
                    **self.entropy_estimator_params,
                )
                self._x_entropy_estimator.fit(x)

                self._y_entropy_estimator = EntropyEstimator(
                    **self.entropy_estimator_params,
                )
                self._y_entropy_estimator.fit(y)
            else:
                raise NotImplementedError
        elif self._x_is_discrete and not self._y_is_discrete:
            self._y_entropy_estimator = EntropyEstimator(
                **self.entropy_estimator_params,
            )
            self._y_entropy_estimator.fit(y)
        elif not self._x_is_discrete and self._y_is_discrete:
            self._x_entropy_estimator = EntropyEstimator(
                **self.entropy_estimator_params,
            )
            self._x_entropy_estimator.fit(x)
        else:
            # No fitting is required when both variables are discrete.
            pass

    def estimate(
        self,
        x: NDArray[Any] | Tensor,
        y: NDArray[Any] | Tensor,
    ) -> tuple[float, float]:
        """Mutual information estimation.

        Args:
            x: I.i.d. samples from X.
            y: I.i.d. samples from Y.

        """
        if x.shape[0] != y.shape[0]:
            msg = "x and y must have the same length"
            raise ValueError(msg)

        if not self._x_is_discrete and not self._y_is_discrete:
            return self._estimate_cont_cont(x, y)
        if self._x_is_discrete and not self._y_is_discrete:
            return self._estimate_cont_disc(y, x, self._y_entropy_estimator)
        if not self._x_is_discrete and self._y_is_discrete:
            return self._estimate_cont_disc(x, y, self._x_entropy_estimator)
        return self._estimate_disc_disc(x, y)

    def fit_estimate(self, x: NDArray[Any] | Tensor, y: NDArray[Any] | Tensor) -> float:
        """Fit the estimator and estimate the mutual information.

        Args:
            x: I.i.d. samples from X.
            y: I.i.d. samples from Y.

        """
        self.fit(x, y)
        est, _ = self.estimate(x, y)
        return est

    def _estimate_cont_cont(
        self,
        x: NDArray | Tensor,
        y: NDArray | Tensor,
    ) -> tuple[float, float]:
        """Mutual information estimation for a pair of continuous random variables.

        Args:
            x: I.i.d. samples from X.
            y: I.i.d. samples from Y.

        """
        h_x, h_x_err = self._x_entropy_estimator.estimate(x)

        h_y, h_y_err = self._y_entropy_estimator.estimate(y)

        h_x_y, h_x_y_err = self._x_y_entropy_estimator.estimate(
            np.concatenate([x.cpu(), y.cpu()], axis=1),
        )

        return (h_x + h_y - h_x_y, h_x_err + h_y_err + h_x_y_err)

    def _estimate_cont_disc(
        self,
        x: NDArray[Any] | Tensor,
        y: NDArray[Any] | Tensor,
        x_entropy_estimator: EntropyEstimator,
    ) -> tuple[float, float]:
        """Mutual information estimation for an absolutely continuous X and discrete Y.

        Args:
            x: I.i.d. samples from X.
            y: I.i.d. samples from Y.
            x_entropy_estimator: Entropy estimator for X.

        """
        h_x, h_x_err = x_entropy_estimator.estimate(x)

        # Empirical frequencies estimation.
        frequencies = y.bincount() / y.shape[0]

        # Conditional entropy estimation.
        h_x_mid_y = {}
        for y_ in range(len(frequencies)):
            x_mid_y = x[y_ == y]

            # For every value of y_, it is required to refit the estimator.
            x_mid_y_entropy_estimator = EntropyEstimator(
                **self.entropy_estimator_params,
            )
            x_mid_y_entropy_estimator.fit(x_mid_y)
            h_x_mid_y[y_] = x_mid_y_entropy_estimator.estimate(x_mid_y)

        # Final conditional entropy estimate.
        cond_h_x = math.fsum(
            [frequencies[y] * h_x_mid_y[y][0] for y in range(len(frequencies))],
        )
        cond_h_x_err = math.fsum(
            [frequencies[y] * h_x_mid_y[y][1] for y in range(len(frequencies))],
        )

        return (h_x - cond_h_x, h_x_err + cond_h_x_err)

    def _estimate_disc_disc(
        self,
        x: NDArray | Tensor,
        y: NDArray | Tensor,
    ) -> tuple[float, float]:
        """Mutual information estimation for a pair of discrete random variables.

        Args:
            x: I.i.d. samples from X.
            y: I.i.d. samples from Y.

        """
        h_x = 0.0
        h_y = 0.0
        h_x_y = 0.0

        frequencies_x = Counter(x)
        for x_ in frequencies_x:
            frequencies_x[x_] /= x.shape[0]
            h_x -= frequencies_x[x_] * np.log(frequencies_x[x])

        frequencies_y = Counter(y)
        for y_ in frequencies_y:
            frequencies_y[y_] /= y.shape[0]
            h_y -= frequencies_y[y_] * np.log(frequencies_y[y_])

        frequencies_x_y = Counter(np.concatenate([x.cpu(), y.cpu()], axis=1))
        for x_y in frequencies_x_y:
            frequencies_x_y[x_y] /= x.shape[0]
            h_x_y -= frequencies_x_y[x_y] * np.log(frequencies_x_y[x_y])

        return (h_x + h_y - h_x_y, 0.0)
