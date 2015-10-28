# regressions.pcr

"""A package which implements Principal Component Regression."""

import random

from .. import *


class PCR_NIPALS:
    """PCR using the NIPALS (Nonlinear Iterative Partial Least Squares)
    algorithm for finding the principal components."""

    def __init__(self, X, Y, g = None, variation_explained = None,
                    max_iterations = DEFAULT_MAX_ITERATIONS,
                    iteration_convergence = DEFAULT_EPSILON):

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same '\
            'number of rows (data samples)')

        if (g == None) and (variation_explained == None):
            raise ParameterError('Must specify either the number of ' \
            'principal components g to use or the proportion of data '\
            'variance that must be explained.')

        if (g != None) and (variation_explained != None):
            raise ParameterError('Cannot specify both the number of ' \
            'principal components g to use or the proportion of data '\
            'variance that must be explained.')

        if variation_explained != None:
            if variation_explained < 0.10 or variation_explained > 0.95:
                raise ParameterError('PCR will not reliably be able '\
                'to use principal components that explain less than '\
                '10% or more than 95% of the variation in the data.')

        if variation_explained != None:
            raise ParameterError('Use of variation_explained not ' \
            'implemented yet!')

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]

        if g != None and g < 1 or g > self.max_rank:
            raise ParameterError('Number of required components ' \
            'specified is impossible.')

        self.X_offset = X.mean(0)
        Xc = X - self.X_offset # Xc is the centred version of X
        self.total_variation = (Xc @ Xc.T).trace()

        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset # Yc is the centred version of Y

        self.components = 0
        T = np.empty((self.data_samples, g)) # Scores
        P = np.empty((self.X_variables, g)) # Loadings
        X_j = Xc

        for j in range(0, g):

            t_j = X_j[:,random.randint(0, self.X_variables-1)]
            iteration_count = 0
            iteration_change = iteration_convergence * 10.0

            while iteration_count < max_iterations and \
                        iteration_change > iteration_convergence:

                p_j = X_j.T @ t_j
                p_j /= np.linalg.norm(p_j, 2) # Normalise p_j vectors

                old_t_j = t_j
                t_j = X_j @ p_j
                iteration_change = linalg.norm(t_j - old_t_j)
                iteration_count += 1

            if iteration_count >= max_iterations:
                break

            X_j = X_j - np.outer(t_j, p_j.T) # Reduce in rank
            T[:,j] = t_j
            P[:,j] = p_j
            self.components += 1

        # Only copy the components actually used
        self.T = T[:,0:self.components]
        self.P = P[:,0:self.components]

        self.eigenvalues = (self.T.T @ self.T).diagonal()

        # Find regression parameters
        self.C = np.diag(1.0 / self.eigenvalues) @ self.T.T @ Yc
        self.PgC = self.P @ self.C

    def variation_explained(self):
        return self.eigenvalues.sum() / self.total_variation

    def prediction(self, Z):
        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the same'\
                'number of variables as the original X data')
            return self.Y_offset + (Z - self.X_offset) @ self.PgC
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the same'\
                'number of variables as the original X data')
            return self.Y_offset + (Z - self.X_offset) @ self.PgC


