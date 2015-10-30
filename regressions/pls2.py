# regressions.pls2

"""A package which implements the Partial Least Squares 2 algorithm."""

import random

from . import *


class PLS2:

    """Regression using the PLS2 algorithm."""

    def __init__(self, X, Y, g,
                 max_iterations=DEFAULT_MAX_ITERATIONS,
                 iteration_convergence=DEFAULT_EPSILON,
                 ignore_failures=True):

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same '
                                 'number of rows (data samples)')

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]

        if g < 1 or g > self.max_rank:
            raise ParameterError('Number of required components '
                                 'specified is impossible.')

        self.X_offset = X.mean(0)
        Xc = X - self.X_offset  # Xc is the centred version of X
        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset  # Yc is the centred version of Y

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
