"""A module which implements the Partial Least Squares 2 algorithm."""

import random

from . import *


class PLS2(RegressionBase):

    """Regression using the PLS2 algorithm.

    The PLS2 algorithm forms a set of new latent variables from the
    provided X and Y data samples based on criteria that balance the need
    to explain the variance within X and Y and the covariance between X
    and Y. Regression is then performed on the latent variables. In
    contrast to PLS1, the PLS2 algorithm handles multi-dimensional Y in
    one pass, taking into account all of the Y variables at once. Due to
    the added complexity relative to PLS1, PLS2 is a non-deterministic
    iterative algorithm comparable to the NIPALS algorithm for PCR.

    Note:
        If ``ignore_failures`` is ``True`` then the resulting object
        may have fewer components than requested if convergence does
        not succeed.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract
        max_iterations (int, optional) : Maximum number of iterations of
            NIPALS to attempt
        iteration_convergence (float, optional): Difference in norm
            between two iterations at which point the iteration will be
            considered to have converged.
        ignore_failures (boolean, optional): Do not raise an error if
            iteration has to be abandoned before the requested number
            of components have been recovered

    Attributes:
        components (int): number of components extracted (=g)
        P (ndarray n x g): Loadings on X (Components extracted from data)
        Q (ndarray m x g): Loadings on Y (Components extracted from data)
        T (ndarray N x g): Scores on X
        U (ndarray N x g): Scores on Y
        W (ndarray n x g): Weight vectors
        C (ndarray g x g): Diagonal matrix of regression coefficients
        B (ndarray n x m): Final regression matrix

    """

    def __init__(self, X, Y, g,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 iteration_convergence=DEFAULT_EPSILON,
                 ignore_failures=True):

        Xc, Yc = super()._prepare_data(X, Y)

        if g < 1 or g > self.max_rank:
            raise ParameterError('Number of required components '
                                 'specified is impossible.')

        W = np.empty((self.X_variables, g))
        T = np.empty((self.data_samples, g))
        Q = np.empty((self.Y_variables, g))
        U = np.empty((self.data_samples, g))
        P = np.empty((self.X_variables, g))
        c = np.empty((g,))

        self.components = 0
        X_j = Xc
        Y_j = Yc

        for j in range(0, g):
            u_j = Y_j[:, random.randint(0, self.Y_variables-1)]

            iteration_count = 0
            iteration_change = iteration_convergence * 10.0

            while iteration_count < max_iterations and \
                    iteration_change > iteration_convergence:

                w_j = X_j.T @ u_j
                w_j /= np.linalg.norm(w_j, 2)

                t_j = X_j @ w_j

                q_j = Y_j.T @ t_j
                q_j /= np.linalg.norm(q_j, 2)

                old_u_j = u_j
                u_j = Y_j @ q_j
                iteration_change = linalg.norm(u_j - old_u_j)
                iteration_count += 1

            if iteration_count >= max_iterations:
                if ignore_failures:
                    break
                else:
                    raise ConvergenceError('PLS2 failed to converge for '
                                           'component: '
                                           '{}'.format(self.components+1))

            W[:, j] = w_j
            T[:, j] = t_j
            Q[:, j] = q_j
            U[:, j] = u_j

            t_dot_t = t_j.T @ t_j
            c[j] = (t_j.T @ u_j) / t_dot_t
            P[:, j] = (X_j.T @ t_j) / t_dot_t
            X_j = X_j - np.outer(t_j, P[:, j].T)
            Y_j = Y_j - c[j] * np.outer(t_j, q_j.T)
            self.components += 1

        # If iteration stopped early because of failed convergence, only
        # the actual components will be copied

        self.W = W[:, 0:self.components]
        self.T = T[:, 0:self.components]
        self.Q = Q[:, 0:self.components]
        self.U = U[:, 0:self.components]
        self.P = P[:, 0:self.components]
        self.C = np.diag(c[0:self.components])

        self.B = self.W @ linalg.inv(self.P.T @ self.W) @ self.C @ self.Q.T

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
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            return self.Y_offset + (Z - self.X_offset).T @ self.B
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            for i in range(0, Z.shape[0]):
                result[i, :] = self.Y_offset + \
                              (Z[i, :] - self.X_offset).T @ self.B
            return result

    def prediction_iterative(self, Z):

        """Predict the output resulting from a given input, iteratively

        This produces the same output as the one-step version ``prediction``
        but works by applying each loading in turn to extract the latent
        variables corresponding to the input.

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
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')

            x_j = Z - self.X_offset
            t = np.empty((self.components))
            for j in range(0, self.components):
                t[j] = x_j @ self.W[:, j]
                x_j = x_j - t[j] * self.P[:, j]
            result = self.Y_offset + t  @ self.C @ self.Q.T

            return result

        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            t = np.empty((self.components))

            for k in range(0, Z.shape[0]):
                x_j = Z[k, :] - self.X_offset
                for j in range(0, self.components):
                    t[j] = x_j @ self.W[:, j]
                    x_j = x_j - t[j] * self.P[:, j]
                result[k, :] = self.Y_offset + t  @ self.C @ self.Q.T

            return result
