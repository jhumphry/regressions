"""A module which implements kernel PLS."""

import random

from . import *


class Kernel_PLS(RegressionBase):

    """Non-linear Kernel PLS regression using the PLS2 algorithm

    This class implements kernel PLS regression by transforming the input
    X data into feature space by applying a kernel function between each
    pair of inputs. The kernel function provided will be called with two
    vectors and should return a float. Kernels should be symmetrical with
    regard to the order in which the vectors are supplied. The PLS2
    algorithm is then applied to the transformed data. The application of
    the kernel function means that non-linear transformations are
    possible.

    Note:
        If ``ignore_failures`` is ``True`` then the resulting object
        may have fewer components than requested if convergence does
        not succeed.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract
        X_kernel (function): Kernel function
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
        X_training_set (ndarray N x n): X calibration data (centred)
        K (ndarray N x N): X calibration data transformed into feature space
        P (ndarray n x g): Loadings on K (Components extracted from data)
        Q (ndarray m x g): Loadings on Y (Components extracted from data)
        T (ndarray N x g): Scores on K
        U (ndarray N x g): Scores on Y
        B_RHS (ndarray n x m): Partial regression matrix

    """

    def __init__(self, X, Y, g, X_kernel,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 iteration_convergence=DEFAULT_EPSILON,
                 ignore_failures=True):

        Xc, Yc = super()._prepare_data(X, Y)

        self.X_training_set = Xc
        self.X_kernel = X_kernel

        K = np.empty((self.data_samples, self.data_samples))
        for i in range(0, self.data_samples):
            for j in range(0, i):
                K[i, j] = X_kernel(Xc[i, :], Xc[j, :])
                K[j, i] = K[i, j]
            K[i, i] = X_kernel(Xc[i, :], Xc[i, :])

        centralizer = (np.identity(self.data_samples)) - \
            (1.0 / self.data_samples) * \
            np.ones((self.data_samples, self.data_samples))
        K = centralizer @ K @ centralizer
        self.K = K

        T = np.empty((self.data_samples, g))
        Q = np.empty((self.Y_variables, g))
        U = np.empty((self.data_samples, g))
        P = np.empty((self.data_samples, g))

        self.components = 0
        K_j = K
        Y_j = Yc

        for j in range(0, g):
            u_j = Y_j[:, random.randint(0, self.Y_variables-1)]

            iteration_count = 0
            iteration_change = iteration_convergence * 10.0

            while iteration_count < max_iterations and \
                    iteration_change > iteration_convergence:

                w_j = K_j @ u_j
                t_j = w_j / np.linalg.norm(w_j, 2)

                q_j = Y_j.T @ t_j

                old_u_j = u_j
                u_j = Y_j @ q_j
                u_j /= np.linalg.norm(u_j, 2)
                iteration_change = linalg.norm(u_j - old_u_j)
                iteration_count += 1

            if iteration_count >= max_iterations:
                if ignore_failures:
                    break
                else:
                    raise ConvergenceError('PLS2 failed to converge for '
                                           'component: '
                                           '{}'.format(self.components+1))

            T[:, j] = t_j
            Q[:, j] = q_j
            U[:, j] = u_j

            P[:, j] = (K_j.T @ w_j) / (w_j @ w_j)
            deflator = (np.identity(self.data_samples) - np.outer(t_j.T, t_j))
            K_j = deflator @ K_j @ deflator
            Y_j = Y_j - np.outer(t_j, q_j.T)
            self.components += 1

        # If iteration stopped early because of failed convergence, only
        # the actual components will be copied

        self.T = T[:, 0:self.components]
        self.Q = Q[:, 0:self.components]
        self.U = U[:, 0:self.components]
        self.P = P[:, 0:self.components]

        self.B_RHS = self.U @ linalg.inv(self.T.T @ self.K @ self.U) @ self.Q.T

    def prediction(self, Z):
        """Predict the output resulting from a given input

        Args:
            Z (ndarray of floats): The input on which to make the
                prediction. A one-dimensional array will be interpreted as
                a single multi-dimensional input unless the number of X
                variables in the calibration data was 1, in which case it
                will be interpreted as a set of inputs. A two-dimensional
                array will be interpreted as one multi-dimensional input
                per row.

        Returns:
            Y (ndarray of floats) : The predicted output - either a one
            dimensional array of the same length as the number of
            calibration Y variables or a two dimensional array with the
            same number of columns as the calibration Y data and one row
            for each input row.
        """

        if len(Z.shape) == 1:
            if self.X_variables == 1:
                Z = Z.reshape((Z.shape[0], 1))
                Kt = np.empty((Z.shape[0], self.data_samples))
            else:
                if Z.shape[0] != self.X_variables:
                    raise ParameterError('Data provided does not have the '
                                         'same number of variables as the '
                                         'original X data')
                Z = Z.reshape((1, Z.shape[0]))
                Kt = np.empty((1, self.data_samples))
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            Kt = np.empty((Z.shape[0], self.data_samples))

        for i in range(0, Z.shape[0]):
            for j in range(0, self.data_samples):
                Kt[i, j] = self.X_kernel(Z[i, :] - self.X_offset,
                                         self.X_training_set[j, :])

        centralizer = (1.0 / self.data_samples) * \
            np.ones((Z.shape[0], self.data_samples))

        Kt = (Kt - centralizer @ self.K) @ \
            (np.identity(self.data_samples) -
                (1.0 / self.data_samples) * np.ones(self.data_samples))

        # Fix centralisation - appears to be necessary but not usually
        # mentioned in papers

        Kt -= Kt.mean(0)

        return self.Y_offset + Kt @ self.B_RHS
