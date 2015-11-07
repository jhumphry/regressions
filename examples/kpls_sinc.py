#!/usr/bin/python3

"""An example of the use of non-linear kernel PLS regression on the output of
a sinc function contaminated by noise.

Reproduces some of the figures from "Kernel Partial Least Squares Regression
in Reproducing Kernel Hilbert Space" by Roman Rosipal and Leonard J Trejo.
Journal of Machine Learning Research 2 (2001) 97-123"""

import math
import random

import numpy as np
import matplotlib.pyplot as plt

from regressions import kernel_pls, kernels

# Perform Kernel PLS on an uncontaminated sinc function to view the principal
# components

x_values = np.linspace(-10.0, 10.0, 100)

pure_sinc = np.sin(np.abs(x_values)) / np.abs(x_values)
pure_sinc -= pure_sinc.mean()

pure_kpls = kernel_pls.Kernel_PLS(X=x_values,
                                  Y=pure_sinc,
                                  g=4,
                                  X_kernel=kernels.std_gaussian)

# Contaminate the sinc function with some Gaussian noise and perform kernel
# PLS on this revised version

noisy_sinc = pure_sinc + np.random.normal(loc=0.0, scale=0.2, size=100)
noisy_sinc -= noisy_sinc.mean()

noisy_kpls = kernel_pls.Kernel_PLS(X=x_values,
                                   Y=noisy_sinc,
                                   g=1,
                                   X_kernel=kernels.std_gaussian)

# Choose some test points and use the kernel PLS results to predict /
# reconstruct the true output of the sinc function

test_x = np.linspace(-10.0, 10.0, 80)
test_y = (np.sin(np.abs(test_x)) / np.abs(test_x))
test_kpls_reconstruction = noisy_kpls.prediction(test_x)

# Perform PLS on the calibration data with a range of numbers of components and
# measure the mean squared error when the test points are predicted

test_mse = np.empty((30,))
for i in range(1, 31):
    test_kpls = kernel_pls.Kernel_PLS(X=x_values,
                                      Y=noisy_sinc,
                                      g=i,
                                      X_kernel=kernels.std_gaussian)
    prediction = test_kpls.prediction(test_x)
    test_mse[i-1] = ((prediction - test_y)**2).mean()

# Plot the results of the above calculations

fig = plt.figure('Kernel PLS applied to sinc function')
fig.set_tight_layout(True)
plt.subplot(3, 1, 1)
plt.title('Principal components found by Gaussian kernel PLS')
plt.plot(x_values, pure_sinc, 'k-', label='$sinc\, x$')
plt.plot(x_values, pure_kpls.P[:, 0], 'r-.', label='1st PC')
plt.plot(x_values, pure_kpls.P[:, 1], 'b--', label='2nd PC')
plt.plot(x_values, pure_kpls.P[:, 2], 'g.', label='3rd PC')
plt.legend()

plt.subplot(3, 1, 2)
plt.title('MSE versus number of components for kernel PLS')
plt.plot(range(1, 31), test_mse, 'b-o')
plt.autoscale(enable=True, axis='x', tight=True)
plt.gca().set_yscale('log', basey=10)

plt.subplot(3, 1, 3)
plt.title('Gaussian kernel PLS reconstruction')
plt.plot(x_values, pure_sinc, 'k--', label='$sinc\, x$')
plt.plot(x_values, noisy_sinc, 'k.', label='$sinc\, x$ with noise')
plt.plot(test_x, test_kpls_reconstruction, 'b-', label='KPLS, 1 PC')
plt.legend()

plt.show()
