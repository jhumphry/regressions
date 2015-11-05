# regressions.kernels

"""A collection of kernels and kernel generators"""

import math

from . import *


def std_gaussian(x, y):
    """A Gaussian kernel with width 1"""

    return 0.3989422804014327 * math.exp(- 0.5 * np.sum((x-y)**2))


def make_gaussian_kernel(width=1.0):
    """Returns a function that implements a Gaussian kernel with the width
    specified"""

    normalization = 1.0 / math.sqrt(2.0 * math.pi * width)
    scale = 1.0 / (2.0 * width**2)

    def gaussian_kernel(x, y):
        return normalization * math.exp(-scale * np.sum((x-y)**2))

    return gaussian_kernel
