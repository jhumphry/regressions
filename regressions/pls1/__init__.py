# regressions.pls1

"""A package which implements the Partial Least Squares 1 algorithm."""

import random

from .. import *


class PLS1:

    """Regression using the PLS1 algorithm."""

    def __init__(self, X, Y, g, epsilon=DEFAULT_EPSILON):

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

        self.W = np.empty((self.Y_variables, self.X_variables, g))
        self.P = np.empty((self.Y_variables, self.X_variables, g))
        self.T = np.empty((self.Y_variables, self.data_samples, g))
        self.c = np.empty((self.Y_variables, g))
        self.b = np.empty((self.Y_variables, self.X_variables))

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
                    print('PLS1 failed at iteration: {}'.format(j))
                    break

                p_j = (X_j.T @ t_j) / tt_j

                X_j = X_j - np.outer(t_j, p_j.T)  # Reduce in rank
                y_j = y_j - t_j * c_j

                self.W[z, :, j] = w_j
                self.P[z, :, j] = p_j
                self.T[z, :, j] = t_j
                self.c[z, j] = c_j
            else:
                # N.B - don't try to find the regression matrix if the
                # iteration failed!
                self.b[z, :] = self.W[z, :, :] @ \
                    linalg.inv(self.P[z, :, :].T @ self.W[z, :, :]) @ \
                    self.c[z, :]

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
