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

Available modules
---------------------
mlr
    Standard Multiple Linear Regression for data with homoskedastic and
    serially uncorrelated errors.
cls
    Classical Least Squares - equivalent to multiple linear regression but
    with the regression computed in reverse (X on Y) and then
    (pseudo-)inverted.
pcr
    Principal Component Regression - based on extracting a limited number
    of components of the X data which best span the variance in X, and
    then regressing Y on only those components. Both iterative (NIPALS)
    and SVD approaches are implemented.
pls1
    Partial Least Squares based on the PLS1 algorithm for use with only
    one Y variable but multiple X variables. Multiple Y variables are
    handled completely independently from each other, without using
    information about correlations. Uses an iterative approach.
pls2
    Partial Least Squares based on the PLS2 algorithm for use with
    multiple X and Y variables simultaneously.  Uses an iterative
    approach.
pls_sb
    Partial Least Squares based on the PLS-SB algorithm. This sets up the
    problem in the same way as the PLS2 algorithm but then solves for the
    eigenvectors directly, with a non-iterative deterministic approach.
kernel_pls
    Transforms the input X data into a higher-dimensional feature space
    using a provided kernel, and then applies the PLS2 algorithm. This
    allows non-linear problems to be addressed.
kernels
    A collection of kernels to use with kernel_pls

"""

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

DEFAULT_MAX_ITERATIONS = 100
"""Default maximum number of iterations that iterative routines will
attempt before raising a ConvergenceError."""

DEFAULT_EPSILON = 0.001
"""A default epsilon value used in various places, such as to decide when
iterations have converged sufficiently."""
