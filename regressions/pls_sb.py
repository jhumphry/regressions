# regressions.pls_sb

"""A package which implements the PLS-SB algorithm."""

import random

from . import *


class PLS_SB:

    """Regression using the PLS-SB algorithm."""

    def __init__(self, X, Y, g):

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

        tmp = Xc.T @ Yc
        XTYYTX = tmp @ tmp.T
        ev, W = linalg.eig(XTYYTX)

        self.components = g
        self.W = W[:, g-1::-1].real

        self.T = Xc @ self.W
        self.Q = Yc.T @ self.T
        self.Q /= np.linalg.norm(self.Q, axis=0)
        self.U = Yc @ self.Q
        t_dot_t = (self.T.T @ self.T).diagonal()
        self.C = np.diag((self.T.T @ self.U).diagonal() / t_dot_t)
        self.P = (Xc.T @ self.T) / t_dot_t

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
