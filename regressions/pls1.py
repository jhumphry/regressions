# regressions.pls1

"""A package which implements the Partial Least Squares 1 algorithm."""

import random

from . import *


class PLS1:

    """Regression using the PLS1 algorithm."""

    def __init__(self, X, Y, g,
                 epsilon=DEFAULT_EPSILON, ignore_failures=False):

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

        self.components = g

        W = np.empty((self.Y_variables, self.X_variables, g))
        P = np.empty((self.Y_variables, self.X_variables, g))
        T = np.empty((self.Y_variables, self.data_samples, g))
        c = np.empty((self.Y_variables, g))
        b = np.empty((self.Y_variables, self.X_variables))

        for z in range(0, self.Y_variables):

            X_j = Xc
            y_j = Yc[:, z]

            for j in range(0, g):

                w_j = X_j.T @ y_j
                w_j /= linalg.norm(w_j, 2)

                t_j = X_j @ w_j
                tt_j = t_j.T @ t_j

                c_j = (t_j.T @ y_j) / tt_j
                if c_j < epsilon:
                    if ignore_failures:
                        if self.components > j:
                            self.components = j  # See comment below
                        break
                    else:
                        raise ConvergenceError('PLS1 failed at iteration: '
                                               'g={}, j={}'.format(g, j))

                p_j = (X_j.T @ t_j) / tt_j

                X_j = X_j - np.outer(t_j, p_j.T)  # Reduce in rank
                y_j = y_j - t_j * c_j

                W[z, :, j] = w_j
                P[z, :, j] = p_j
                T[z, :, j] = t_j
                c[z, j] = c_j
            else:
                # N.B - don't try to find the regression matrix if the
                # iteration failed! Inversion won't work...
                b[z, :] = W[z, :, :] @ \
                    linalg.inv(P[z, :, :].T @ W[z, :, :]) @ \
                    c[z, :]

        # If one of the iterations fails due to c_j becoming too small, then
        # self.components will be reduced and the output will be cut down to
        # the lowest number of iterations achieved for any of the Y variables.
        # Of course, b may no longer be a particularly good regression vector
        # in this case.
        self.W = W[:, :, 0:self.components]
        self.P = P[:, :, 0:self.components]
        self.T = T[:, :, 0:self.components]
        self.c = c[:, 0:self.components]
        self.b = b

    def prediction(self, Z):
        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            return self.Y_offset + (Z - self.X_offset).T @ self.b.T
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            for i in range(0, Z.shape[0]):
                result[i, :] = self.Y_offset + \
                              (Z[i, :] - self.X_offset).T @ self.b.T
            return result

    def prediction_iterative(self, Z):
        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = self.Y_offset.copy()
            for k in range(0, self.Y_variables):
                x_j = Z - self.X_offset
                t = np.empty((self.components))
                for j in range(0, self.components):
                    t[j] = x_j @ self.W[k, :, j]
                    x_j = x_j - t[j] * self.P[k, :, j]
                result[k] += self.c[k, :] @ t

            return result

        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            result[:, :] = self.Y_offset.copy()
            for l in range(0, Z.shape[0]):
                for k in range(0, self.Y_variables):
                    x_j = Z[l, :] - self.X_offset
                    t = np.empty((self.components))
                    for j in range(0, self.components):
                        t[j] = x_j @ self.W[k, :, j]
                        x_j = x_j - t[j] * self.P[k, :, j]
                    result[l, k] += self.c[k, :] @ t

            return result
