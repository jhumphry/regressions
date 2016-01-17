#!/usr/bin/python3

"""An example of the use of non-linear kernel PLS regression on the output of
a function z(x) = 4.26(exp (−x) − 4 exp (−2x) + 3 exp (−3x))

Reproduces figure 3 from "Overview and Recent Advances in Partial Least
Squares" Roman Rosipal and Nicole Krämer SLSFS 2005, LNCS 3940, pp. 34–51,
2006 and figure 3 from "Nonlinear Partial Least Squares: An Overview" Roman
Rosipal """

# Copyright (c) 2015, James Humphry - see LICENSE file for details

import numpy as np
import matplotlib.pyplot as plt

from regressions import kernel_pls, kernels


def z(x):
    '''Example function'''
    return 4.26 * (np.exp(-x) - 4 * np.exp(-2.0*x) + 3 * np.exp(-3.0*x))

# Define the kernel

kern = kernels.make_gaussian_kernel(width=1.8)

# Create sample data

x_values = np.linspace(0.0, 3.5, 100)

z_pure = z(x_values)
z_pure -= z_pure.mean(0)  # Ensure z_pure is centered

noise = np.random.normal(loc=0.0, scale=0.2, size=100)
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
fig.clear()

# Plot some of the extracted components

# These figures plot the underlying function based on 100 (xi, z(xi)) pairs
# as a dotted line in the original problem space. The component extracted
# is a single vector in the 100-dimensional transformed feature space. Each
# dimension in feature space corresponds to a K(?, xi) kernel function. As
# the kernel in this case is the Gaussian kernel which is spacially
# localised, it is workable to map each K(?, xi) function to the
# x-cordinate xi for display in this manner. In the general case,
# meaningfully plotting the components in kernel space is likely to be
# difficult.

fig = plt.figure('Components found in Kernel PLS regression')

fig.set_tight_layout(True)

for i in range(0, 8):
    plt.subplot(4, 2, (i+1))
    plt.title('Kernel PLS component {}'.format((i+1)))
    plt.plot(x_values, z_pure, 'k--')
    plt.plot(x_values, kpls_8.P[:, i], 'k-')
    plt.gca().set_ybound(lower=-1.5, upper=1.0)

plt.show()
fig.clear()
