# regressions.cls

"""A package which implements Classical Least Squares Regression."""

import random

from . import *


class CLS:

    """Classical Least Squares Regression"""

    def __init__(self, X, Y):

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same number of '
                                 'rows (data samples)')

        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))

        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))

        if Y.shape[0] <= Y.shape[1]:
            raise ParameterError('CLS requires more rows (data samples) than '
                                 'output variables (columns of Y data)')

        if X.shape[1] < Y.shape[1]:
            raise ParameterError('CLS requires at least as input variables '
                                 '(columns of X data) as output variables '
                                 '(columns of Y data)')

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]

        self.X_offset = X.mean(0)
        Xc = X - self.X_offset  # Xc is the centred version of X
        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset  # Yc is the centred version of Y

        self.A = linalg.inv(Yc.T @ Yc) @ Yc.T @ Xc
        self.A_pinv = self.A.T @ linalg.inv(self.A @ self.A.T)

    def prediction(self, Z):
        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the same '
                                     'number of variables as the original X '
                                     'data')
            return self.Y_offset + (Z - self.X_offset) @ self.A_pinv
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            for i in range(0, Z.shape[0]):
                result[i, :] = self.Y_offset + (Z[i, :] - self.X_offset) \
                    @ self.A_pinv
            return result
