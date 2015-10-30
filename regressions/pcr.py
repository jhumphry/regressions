# regressions.pcr

"""A package which implements Principal Component Regression."""

import random

from . import *


class PCR_NIPALS:

    """PCR using the NIPALS (Nonlinear Iterative Partial Least Squares)
    algorithm for finding the principal components."""

    def __init__(self, X, Y, g=None, variation_explained=None,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 iteration_convergence=DEFAULT_EPSILON,
                 ignore_failures=True):

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same '
                                 'number of rows (data samples)')

        if (g is None) == (variation_explained is None):
            raise ParameterError('Must specify either the number of '
                                 'principal components g to use or the '
                                 'proportion of data variance that '
                                 'must be explained.')

        if variation_explained is not None:
            if variation_explained < 0.001 or\
                    variation_explained > 0.999:
                raise ParameterError('PCR will not reliably be able to '
                                     'use principal components that '
                                     'explain less than 0.1% or more '
                                     'than 99.9% of the variation in '
                                     'the data.')

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]

        if g is not None:
            if g < 1 or g > self.max_rank:
                raise ParameterError('Number of required components '
                                     'specified is impossible.')

        self.X_offset = X.mean(0)

        self._perform_pca(X, g, variation_explained,
                          max_iterations, iteration_convergence,
                          ignore_failures)

        # Find regression parameters
        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset  # Yc is the centred version of Y

        self.C = np.diag(1.0 / self.eigenvalues) @ self.T.T @ Yc
        self.PgC = self.P @ self.C

    def _perform_pca(self, X, g=None, variation_explained=None,
                     max_iterations=DEFAULT_MAX_ITERATIONS,
                     iteration_convergence=DEFAULT_EPSILON,
                     ignore_failures=True):

        """A non-public routine that performs the PCA using an
        appropriate method and sets up self.total_variation, self.T,
        self.P, self.eignvalues and self.components."""

        Xc = X - self.X_offset  # Xc is the centred version of X
        self.total_variation = (Xc @ Xc.T).trace()

        T = np.empty((self.data_samples, self.max_rank))  # Scores
        P = np.empty((self.X_variables, self.max_rank))  # Loadings
        eig = np.empty((self.max_rank,))

        self.components = 0
        X_j = Xc

        while True:

            t_j = X_j[:, random.randint(0, self.X_variables-1)]
            iteration_count = 0
            iteration_change = iteration_convergence * 10.0

            while iteration_count < max_iterations and\
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
        return self.eigenvalues.sum() / self.total_variation

    def prediction(self, Z):
        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the '
                                     'same number of variables as the '
                                     'original X data')
            return self.Y_offset + (Z - self.X_offset) @ self.PgC
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the '
                                     'same number of variables as the '
                                     'original X data')
            return self.Y_offset + (Z - self.X_offset) @ self.PgC


class PCR_SVD(PCR_NIPALS):

    """PCR using the SVD method for finding the principal components."""

    def _perform_pca(self, X, g=None, variation_explained=None,
                     max_iterations=DEFAULT_MAX_ITERATIONS,
                     iteration_convergence=DEFAULT_EPSILON):

        """A non-public routine that performs the PCA using an
        appropriate method and sets up self.total_variation, self.T,
        self.P, self.eignvalues and self.components."""

        Xc = X - self.X_offset  # Xc is the centred version of X
        self.total_variation = (Xc @ Xc.T).trace()

        u, s, v = linalg.svd(Xc, full_matrices=False)

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
