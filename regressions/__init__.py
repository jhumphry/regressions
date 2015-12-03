"""
Regressions
===========

Provides various forms of regression which are not all covered by other
Python statistical packages. The aim is to achieve clarity of
implementation with speed a secondary goal. Python 3.5 and Numpy 1.10 or
greater are required as the new '@' matrix multiplication operator is
used.

All of the regressions require the X and Y data to be provided in the form
of matrices, with one row per data sample and the same number of data
samples in each. Currently missing data or NaN are not supported.

"""

# Copyright (c) 2015, James Humphry - see LICENSE file for details

import abc

import numpy as np
try:
    import scipy.linalg as linalg
    _linalg_source = 'scipy'
except ImportError:
    import numpy.linalg as linalg
    _linalg_source = 'numpy'


class ParameterError(Exception):
    """Parameters passed to a regression routine are unacceptable

    This is a generic exception used to indicate that the parameters
    passed are mis-matched, nonsensical or otherwise problematic.
    """
    pass


class ConvergenceError(Exception):
    """Iterative algorithm failed to converge.

    Many of the routines used for regressions are iterative and in some
    cases may not converge. This is mainly likely to happen if the data
    has pathological features, or if too many components of a data set
    have been extracted by an iterative process and the residue is
    becoming dominated by rounding or other errors.
    """
    pass

DEFAULT_MAX_ITERATIONS = 250
"""Default maximum number of iterations that iterative routines will
attempt before raising a ConvergenceError."""

DEFAULT_EPSILON = 1.0E-6
"""A default epsilon value used in various places, such as to decide when
iterations have converged sufficiently."""


class RegressionBase(metaclass=abc.ABCMeta):

    """Abstract base class for regressions

    All the various types of regression objects will have at least the
    attributes present here.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        standardize_X (boolean, optional): Standardize the X data
        standardize_Y (boolean, optional): Standardize the Y data

    Attributes:
        data_samples (int): number of calibration data samples (=N)
        max_rank (int): maximum rank of calibration X-data (limits the
            number of components that can be found)
        X_variables (int): number of X variables (=n)
        Y_variables (int): number of Y variables (=m)
        X_offset (float): Offset of calibration X data from zero
        Y_offset (float): Offset of calibration Y data from zero
        standardized_X (boolean): whether X data had variance standardized
        standardized_Y (boolean): whether Y data had variance standardized
        X_rscaling (float): the reciprocal of the scaling factor used for X
        Y_scaling (float): the scaling factor used for Y
    """

    @abc.abstractmethod
    def __init__(self, X, Y, standardize_X=False, standardize_Y=False):
        pass

    def _prepare_data(self, X, Y, standardize_X=False, standardize_Y=False):

        """A private method that conducts routine data preparation

        Sets all of the RegressionBase attributes on ``self`` and returns
        suitably centred and (where requested) variance standardized X and
        Y data.

        Args:
            X (ndarray N x n): X calibration data, one row per data sample
            Y (ndarray N x m): Y calibration data, one row per data sample
            standardize_X (boolean, optional): Standardize the X data
            standardize_Y (boolean, optional): Standardize the Y data

        Returns:
            Xc (ndarray N x n): Centralized and standardized X data
            Yc (ndarray N x m): Centralized and standardized Y data

        """

        if X.shape[0] != Y.shape[0]:
            raise ParameterError('X and Y data must have the same number of '
                                 'rows (data samples)')

        # Change 1-D arrays into column vectors
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0], 1))

        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))

        self.max_rank = min(X.shape)
        self.data_samples = X.shape[0]
        self.X_variables = X.shape[1]
        self.Y_variables = Y.shape[1]
        self.standardized_X = standardize_X
        self.standardized_Y = standardize_Y

        self.X_offset = X.mean(0)
        Xc = X - self.X_offset

        if standardize_X:
            # The reciprocals of the standard deviations of each column are
            # stored as these are what are needed for fast prediction
            self.X_rscaling = 1.0 / Xc.std(0, ddof=1)
            Xc *= self.X_rscaling
        else:
            self.X_rscaling = None

        self.Y_offset = Y.mean(0)
        Yc = Y - self.Y_offset
        if standardize_Y:
            self.Y_scaling = Y.std(0, ddof=1)
            Yc /= self.Y_scaling
        else:
            self.Y_scaling = None

        return Xc, Yc
