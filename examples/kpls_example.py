#!/usr/bin/python3

"""An example of the use of non-linear kernel PLS regression on the output of
a function z(x) = 4.26(exp (−x) − 4 exp (−2x) + 3 exp (−3x))

Reproduces figure 3 from "Overview and Recent Advances in Partial Least
Squares" Roman Rosipal and Nicole Krämer SLSFS 2005, LNCS 3940, pp. 34–51,
2006. """

import math
import random

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")

from regressions import kernel_pls, kernels


def z(x):
    return 4.26 * (np.exp(-x) - 4 * np.exp(-2.0*x) + 3 * np.exp(-3.0*x))

# Define the kernel

kern = kernels.make_gaussian_kernel(width=1.8)

# Create sample data

x_values = np.linspace(0.0, 3.5, 100).reshape((100, 1))

z_pure = z(x_values)
z_pure -= z_pure.mean(0)  # Ensure z_pure is centered

noise = np.random.normal(loc=0.0, scale=0.2, size=100).reshape((100, 1))
z_noisy = z_pure + noise
z_noisy -= z_noisy.mean(0)  # Ensure z_noisy is centered

# Perform Kernel PLS

kpls_1 = kernel_pls.Kernel_PLS(X=x_values,
                               Y=z_noisy,
                               g=1,
                               X_kernel=kern)

kpls_1_results = kpls_1.prediction(x_values)

kpls_4 = kernel_pls.Kernel_PLS(X=x_values,
                               Y=z_noisy,
                               g=4,
                               X_kernel=kern)

kpls_4_results = kpls_4.prediction(x_values)

kpls_8 = kernel_pls.Kernel_PLS(X=x_values,
                               Y=z_noisy,
                               g=8,
                               X_kernel=kern)

kpls_8_results = kpls_8.prediction(x_values)

# Plot the results of the above calculations

fig = plt.figure('An example of Kernel PLS regression')

plt.title('An example of Kernel PLS regression')
plt.plot(x_values, z_pure, 'k-', label='$z(.)$')
plt.plot(x_values, z_noisy, 'k+', label='$z(.)$ with noise')
plt.plot(x_values, kpls_1_results, 'k--', label='KPLS 1C')
plt.plot(x_values, kpls_4_results, 'k:', label='KPLS 4C')
plt.plot(x_values, kpls_8_results, 'k-.', label='KPLS 8C')
plt.legend(loc=4)

plt.show()
