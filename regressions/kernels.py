"""A collection of kernels and kernel generators

These are mainly for use in kernel PLS. All of the kernels have the form
K(x, y) where x and y are either floats or numpy.ndarray of float.
"""

import math

from . import *
from typing import Union
from collections.abc import Callable

Kernel_Function = Callable[[Union[float, np.ndarray],
                            Union[float, np.ndarray]], float]

def std_gaussian(x : Union[float, np.ndarray],
                 y : Union[float, np.ndarray]) -> float:
    """A Gaussian kernel with width 1.

    The Gaussian kernel with standard deviation 1 is a routine choice.

    Args:
        x (float or numpy.ndarray of float): The x coordinate
        y (float or numpy.ndarray of float): The y coordinate
    """

    return 0.3989422804014327 * math.exp(- 0.5 * np.sum((x-y)**2))


def make_gaussian_kernel(width : float=1.0) -> Kernel_Function:
    """Create a Gaussian kernel with adjustable width

    Args:
        width (float) : The standard deviation of the Gaussian function
            which adjusts the width of the resulting kernel.

    Returns:
        gaussian_kernel (function) : A function of two floats or
        numpy.ndarray of floats which computes the Gaussian kernel of
        the desired width.
    """

    normalization = 1.0 / math.sqrt(2.0 * math.pi * width)
    scale = 1.0 / (2.0 * width**2)

    def gaussian_kernel(x, y):
        return normalization * math.exp(-scale * np.sum((x-y)**2))

    return gaussian_kernel
