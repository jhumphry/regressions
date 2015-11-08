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
