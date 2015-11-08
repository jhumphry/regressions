Introduction
============

This package provides various forms of regression. The aim of these modules is
to achieve clarity of implementation with a clear connection to the
mathematical descriptions of the algorithms. The motivation for creating the
package was the desire to learn about and explore the use of Principal
Components Regression, Partial Least Squares regression and non-linear
kernel-based Partial Least Squares regression.

Python 3.5 and Numpy 1.10 or greater are required as the new '@' matrix
multiplication operator is used. If SciPy is available some linear algebra
routines may be used as they can sometimes be faster than the routines in
Numpy - however SciPy is not required. Matplotlib is used by the examples to
display the results.

Overview of modules available
-----------------------------

:py:mod:`regressions.mlr`
    Standard Multiple Linear Regression for data with homoskedastic and
    serially uncorrelated errors.
:py:mod:`regressions.cls`
    Classical Least Squares - equivalent to multiple linear regression but
    with the regression computed in reverse (X on Y) and then
    (pseudo-)inverted.
:py:mod:`regressions.pcr`
    Principal Component Regression - based on extracting a limited number
    of components of the X data which best span the variance in X, and
    then regressing Y on only those components. Both iterative (NIPALS)
    and SVD approaches are implemented.
:py:mod:`regressions.pls1`
    Partial Least Squares based on the PLS1 algorithm for use with only
    one Y variable but multiple X variables. Multiple Y variables are
    handled completely independently from each other, without using
    information about correlations. Uses an iterative approach.
:py:mod:`regressions.pls2`
    Partial Least Squares based on the PLS2 algorithm for use with
    multiple X and Y variables simultaneously.  Uses an iterative
    approach.
:py:mod:`regressions.pls_sb`
    Partial Least Squares based on the PLS-SB algorithm. This sets up the
    problem in the same way as the PLS2 algorithm but then solves for the
    eigenvectors directly, with a non-iterative deterministic approach.
:py:mod:`regressions.kernel_pls`
    Transforms the input X data into a higher-dimensional feature space
    using a provided kernel, and then applies the PLS2 algorithm. This
    allows non-linear problems to be addressed.
:py:mod:`regressions.kernels`
    A collection of kernels to use with kernel_pls
