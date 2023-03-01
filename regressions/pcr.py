"""A module which implements Principal Component Regression."""

import random

from . import *


class PCR_NIPALS(RegressionBase):

    """Principal Components Regression using the NIPALS algorithm

    PCR forms a set of new latent variables from the provided X data
    samples which describe as much of the variance in the X data as
    possible. The latent variables are then regressed against the provided
    Y data. PCR is connected with Principal Components Analysis, where the
    latent variables are referred to as Principal Components.

    This class uses the Non-linear Iterative PArtial Least Squares
    algorithm to extract the components. Either a fixed number of
    components should be specified using the ``g`` argument, or a target
    proportion of variation explained by the components should be
    specified via ``variation_explained``. The variables of the X and Y
    data can have their variances standardized. This is useful if they are
    of heterogeneous types as otherwise the components extracted can be
    dominated by the effects of different measurement scales rather than
    by the actual data.

    Note:
        If ``ignore_failures`` is ``True`` then the resulting object
        may have fewer components than requested if convergence does
        not succeed.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract
        variation_explained (float): Proportion of variance in X
            calibration data that the components extracted should explain
            (from 0.001 - 0.999)
        standardize_X (boolean, optional): Standardize the X data
        standardize_Y (boolean, optional): Standardize the Y data
        max_iterations (int, optional) : Maximum number of iterations of
            NIPALS to attempt
        iteration_convergence (float, optional): Difference in norm
            between two iterations at which point the iteration will be
            considered to have converged.
        ignore_failures (boolean, optional): Do not raise an error if
            iteration has to be abandoned before the requested number
            of or coverage by components has been achieved.

    Attributes:
        components (int): number of components extracted (=g)
        T (ndarray N x g): Scores
        P (ndarray n x g): Loadings (Components extracted from data)
        eigenvalues (ndarray g): Eigenvalues extracted
        total_variation (float): Total variation in calibration X data
        C (ndarray g x m): Regression coefficients
        PgC (ndarray n x m): Precalculated matrix product of P (limited to
            g components) and C

    """

    def __init__(self, X, Y, g=None, variation_explained=None,
                 standardize_X=False, standardize_Y=False,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 iteration_convergence=DEFAULT_EPSILON,
                 ignore_failures=True):

        if max_iterations < 1:
            raise ParameterError("At least one iteration is necessary")

        if iteration_convergence <= 0.0:
            raise ParameterError("Iteration convergence limit must be positive")

        if (g is None) == (variation_explained is None):
            raise ParameterError('Must specify either the number of principal '
                                 'components g to use or the proportion of '
                                 'data variance that must be explained.')

        if variation_explained is not None:
            if variation_explained < 0.001 or\
                    variation_explained > 0.999:
                raise ParameterError('PCR will not reliably be able to use '
                                     'principal components that explain less '
                                     'than 0.1% or more than 99.9% of the '
                                     'variation in the data.')

        Xc, Yc = super()._prepare_data(X, Y, standardize_X, standardize_Y)

        if g is not None:
            if g < 1 or g > self.max_rank:
                raise ParameterError('Number of required components specified '
                                     'is impossible.')

        if standardize_X:
            self.total_variation = self.X_variables * (self.data_samples - 1.0)
        else:
            self.total_variation = (Xc @ Xc.T).trace()

        self._perform_pca(Xc, g, variation_explained,
                          max_iterations, iteration_convergence,
                          ignore_failures)

        # Find regression parameters
        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset
        if standardize_Y:
            self.Y_scaling = Y.std(0, ddof=1)
            Yc /= self.Y_scaling
        else:
            self.Y_scaling = None

        self.C = np.diag(1.0 / self.eigenvalues) @ self.T.T @ Yc
        self.PgC = self.P @ self.C

    def _perform_pca(self, X, g=None, variation_explained=None,
                     max_iterations=DEFAULT_MAX_ITERATIONS,
                     iteration_convergence=DEFAULT_EPSILON,
                     ignore_failures=True):

        """A non-public routine that performs the PCA using an appropriate
        method and sets up self.T, self.P, self.eignvalues and
        self.components."""

        T = np.empty((self.data_samples, self.max_rank))  # Scores
        P = np.empty((self.X_variables, self.max_rank))  # Loadings
        eig = np.empty((self.max_rank,))

        self.components = 0
        X_j = X

        while True:

            t_j = X_j[:, random.randint(0, self.X_variables-1)]
            iteration_count = 0
            iteration_change = iteration_convergence * 10.0

            while iteration_count < max_iterations and \
                    iteration_change > iteration_convergence:

                p_j = X_j.T @ t_j
                p_j /= np.linalg.norm(p_j, 2)  # Normalise p_j vectors

                old_t_j = t_j
                t_j = X_j @ p_j
                iteration_change = linalg.norm(t_j - old_t_j)
                iteration_count += 1

            if iteration_count >= max_iterations:
                if ignore_failures:
                    break
                else:
                    raise ConvergenceError('NIPALS PCA for PCR failed to '
                                           'converge for component: '
                                           '{}'.format(self.components+1))

            X_j = X_j - np.outer(t_j, p_j.T)  # Reduce in rank
            T[:, self.components] = t_j
            P[:, self.components] = p_j
            eig[self.components] = t_j @ t_j
            self.components += 1

            if g is not None:
                if self.components == g:
                    break

            if variation_explained is not None:
                if eig[0:self.components].sum() >= \
                   variation_explained * self.total_variation:
                    break

        # Only copy the components actually used
        self.T = T[:, 0:self.components]
        self.P = P[:, 0:self.components]

        self.eigenvalues = eig[0:self.components]

    def variation_explained(self):

        """Return the proportion of variation explained

        Returns:
            variation_explained (float): Proportion of the total variation
            in the X data explained by the extracted principal components.

        """

        return self.eigenvalues.sum() / self.total_variation

    def prediction(self, Z):

        """Predict the output resulting from a given input

        Args:
            Z (ndarray of floats): The input on which to make the
                prediction. Must either be a one dimensional array of the
                same length as the number of calibration X variables, or a
                two dimensional array with the same number of columns as
                the calibration X data and one row for each input row.

        Returns:
            Y (ndarray of floats) : The predicted output - either a one
            dimensional array of the same length as the number of
            calibration Y variables or a two dimensional array with the
            same number of columns as the calibration Y data and one row
            for each input row.
        """

        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the same '
                                     'number of variables as the original X '
                                     'data')
        elif Z.shape[1] != self.X_variables:
            raise ParameterError('Data provided does not have the same '
                                 'number of variables as the original X data')

        tmp = (Z - self.X_offset)
        if self.standardized_X:
            tmp *= self.X_rscaling
        tmp = tmp @ self.PgC
        if self.standardized_Y:
            tmp *= self.Y_scaling
        return self.Y_offset + tmp


class PCR_SVD(PCR_NIPALS):

    """Principal Components Regression using SVD

    This class implements PCR with the same mathematical goals as
    :py:class:`PCR_NIPALS` but using a different method to extract the
    principal components. The convergence criteria in the NIPALS algorithm
    can be formulated into an eigenvalue problem and solved directly using
    an existing SVD-based solver. This has the advantage of being entirely
    deterministic, but the disadvantage that all components have to be
    extracted each time, even if only a few are required to explain most
    of the variance in X.

    Note:
        The attributes of the resulting class are exactly the same as for
        :py:class:`PCR_NIPALS`.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract
        variation_explained (float): Proportion of variance in X
            calibration data that the components extracted should explain
            (from 0.001 - 0.999)
        standardize_X (boolean, optional): Standardize the X data
        standardize_Y (boolean, optional): Standardize the Y data
        max_iterations  : Not relevant for SVD
        iteration_convergence : Not relevant for SVD
        ignore_failures: Not relevant for SVD

    """

    def _perform_pca(self, X, g=None, variation_explained=None,
                     max_iterations=DEFAULT_MAX_ITERATIONS,
                     iteration_convergence=DEFAULT_EPSILON,
                     ignore_failures=True):

        """A non-public routine that performs the PCA using an appropriate
        method and sets up self.total_variation, self.T, self.P,
        self.eignvalues and self.components."""

        u, s, v = linalg.svd(X, full_matrices=False)

        T = u @ np.diag(s)
        P = v.T
        eig = (T.T @ T).diagonal()

        if g is not None:
            self.T = T[:, 0:g]
            self.P = P[:, 0:g]
            self.eigenvalues = eig[0:g]
            self.components = g
        else:
            cuml = (eig.cumsum()/self.total_variation)
            self.components = cuml.searchsorted(variation_explained) + 1
            self.T = T[:, 0:self.components]
            self.P = P[:, 0:self.components]
            self.eigenvalues = eig[0:self.components]
