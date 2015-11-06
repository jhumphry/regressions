"""A module which implements Multiple Linear Regression."""

import random

from . import *


class MLR:

    """Multiple Linear Regression

    Standard multiple linear regression assumes the relationship between the
    variables (once the means have been subtracted to center both variables)
    is Y = A X + E where E is a vector of zero-mean noise vectors.

    Note :
        The regression matrix B is found using the pseudo-inverse. In
        order for this to be calculable, the number of calibration samples
        ``N`` has be be larger than the number of X variables ``n``, and
        there must not be any collinearities in the calibration X data.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample

    Attributes:
        data_samples (int): number of calibration data samples (=N)
        max_rank (int): maximum rank of calibration X-data
        X_variables (int): number of X variables (=n)
        Y_variables (int): number of Y variables (=m)
        X_offset (float): Offset of calibration X data from zero
        Y_offset (float): Offset of calibration Y data from zero
            components (int): number of components extracted (=g)
        B (ndarray m x n): Resulting regression matrix

    """

    def __init__(self, X, Y):

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same number of '
                                 'rows (data samples)')

        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))

        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))

        if X.shape[0] <= X.shape[1]:
            raise ParameterError('MLR requires more rows (data samples) than '
                                 'input variables (columns of X data)')

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]

        self.X_offset = X.mean(0)
        Xc = X - self.X_offset  # Xc is the centred version of X
        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset  # Yc is the centred version of Y

        self.B = linalg.inv(Xc.T @ Xc) @ Xc.T @ Yc

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
            return self.Y_offset + (Z - self.X_offset) @ self.B
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            for i in range(0, Z.shape[0]):
                result[i, :] = self.Y_offset + (Z[i, :] - self.X_offset) \
                    @ self.B
            return result
