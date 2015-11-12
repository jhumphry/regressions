"""A module which implements Classical Least Squares Regression."""

import random

from . import *


class CLS(RegressionBase):

    """Classical Least Squares Regression

    The classical least squares regression approach is to initially swap the
    roles of the X and Y variables, perform linear regression and then to
    invert the result. It is useful when the number of X variables is larger
    than the number of calibration samples available, when conventional
    multiple linear regression would be unable to proceed.

    Note :
        The regression matrix A_pinv is found using the pseudo-inverse. In
        order for this to be calculable, the number of calibration samples
        ``N`` has be be larger than the number of Y variables ``m``, the
        number of X variables ``n`` must at least equal the number of Y
        variables, there must not be any collinearities in the calibration Y
        data and Yt X must be non-singular.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample

    Attributes:
        A (ndarray m x n): Resulting regression matrix of X on Y
        A_pinv (ndarray m x n): Pseudo-inverse of A

    """

    def __init__(self, X, Y):

        Xc, Yc = super()._prepare_data(X, Y)

        if Yc.shape[0] <= Yc.shape[1]:
            raise ParameterError('CLS requires more rows (data samples) than '
                                 'output variables (columns of Y data)')

        if Xc.shape[1] < Yc.shape[1]:
            raise ParameterError('CLS requires at least as input variables '
                                 '(columns of X data) as output variables '
                                 '(columns of Y data)')

        self.A = linalg.inv(Yc.T @ Yc) @ Yc.T @ Xc
        self.A_pinv = self.A.T @ linalg.inv(self.A @ self.A.T)

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
