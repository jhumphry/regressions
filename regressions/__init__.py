# regressions

"""A package which implements various forms of regression."""

import numpy as np
try:
    import scipy.linalg as linalg
    linalg_source = 'scipy'
except ImportError:
    import numpy.linalg as linalg
    linalg_source = 'numpy'


class ParameterError(Exception):
    """Parameters passed to a regression routine are unacceptable"""
    pass


class ConvergenceError(Exception):
    """Iterative algorithm failed to converge"""
    pass

# Maximum iterations that will be attempted by iterative routines by
# default
DEFAULT_MAX_ITERATIONS = 100

# A default epsilon value used in various places, such as to decide when
# iterations have converged
DEFAULT_EPSILON = 0.001
