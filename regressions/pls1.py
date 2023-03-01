"""A module which implements the Partial Least Squares 1 algorithm."""

from . import *


class PLS1(RegressionBase):

    """Regression using the PLS1 algorithm.

    The PLS1 algorithm forms a set of new latent variables from the
    provided X and Y data samples based on criteria that balance the need
    to explain the variance within X and Y and the covariance between X
    and Y. Regression is then performed on the latent variables. PLS1 only
    addresses the case of a single Y variable and if more than one output
    variable is required then PLS1 will be run multiple times. PLS1 is a
    deterministic algorithm that requires one iteration per component
    extracted.

    Note:
        If ``ignore_failures`` is ``True`` then the resulting object
        may have fewer components than requested if convergence does
        not succeed.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract
        epsilon (float, optional): Value at which the components
            extracted will be considered to be too small to be stable
            and iteration will cease
        ignore_failures (boolean, optional): Do not raise an error if
            iteration has to be abandoned before the requested number
            of components have been recovered

    Attributes:
        components (int): number of components extracted (=g)
        W (ndarray m x n x g): Weight vectors
        P (ndarray m x n x g): Loadings (Components extracted from data)
        T (ndarray m x N x g): Scores
        c (ndarray m x g): Regression coefficients
        b (ndarray m x n): Resulting regression matrix

    """

    # Type declarations for attributes:
    components : int
    W : np.ndarray
    P : np.ndarray
    T : np.ndarray
    c : np.ndarray
    b : np.ndarray

    def __init__(self, X : np.ndarray, Y : np.ndarray, g : int,
                 epsilon : float=DEFAULT_EPSILON,
                 ignore_failures : bool=False) -> None:

        if epsilon <= 0.0:
            raise ParameterError("Epsilon must be positive")

        Xc, Yc = super()._prepare_data(X, Y)

        if g < 1 or g > self.max_rank:
            raise ParameterError('Number of required components '
                                 'specified is impossible.')
        self.components = g

        W = np.empty((self.Y_variables, self.X_variables, g))
        P = np.empty((self.Y_variables, self.X_variables, g))
        T = np.empty((self.Y_variables, self.data_samples, g))
        c = np.empty((self.Y_variables, g))
        b = np.empty((self.Y_variables, self.X_variables))

        for z in range(0, self.Y_variables):

            X_j = Xc
            y_j = Yc[:, z]

            for j in range(0, g):

                w_j = X_j.T @ y_j
                w_j /= linalg.norm(w_j, 2)

                t_j = X_j @ w_j
                tt_j = t_j.T @ t_j

                c_j = (t_j.T @ y_j) / tt_j
                if c_j < epsilon:
                    if ignore_failures:
                        if self.components > j:
                            self.components = j  # See comment below
                        break
                    else:
                        raise ConvergenceError('PLS1 failed at iteration: '
                                               'g={}, j={}'.format(g, j))

                p_j = (X_j.T @ t_j) / tt_j

                X_j = X_j - np.outer(t_j, p_j.T)  # Reduce in rank
                y_j = y_j - t_j * c_j

                W[z, :, j] = w_j
                P[z, :, j] = p_j
                T[z, :, j] = t_j
                c[z, j] = c_j
            else:
                # N.B - don't try to find the regression matrix if the
                # iteration failed! Inversion won't work...
                b[z, :] = W[z, :, :] @ \
                    linalg.inv(P[z, :, :].T @ W[z, :, :]) @ \
                    c[z, :]

        # If one of the iterations fails due to c_j becoming too small, then
        # self.components will be reduced and the output will be cut down to
        # the lowest number of iterations achieved for any of the Y variables.
        # Of course, b may no longer be a particularly good regression vector
        # in this case.
        self.W = W[:, :, 0:self.components]
        self.P = P[:, :, 0:self.components]
        self.T = T[:, :, 0:self.components]
        self.c = c[:, 0:self.components]
        self.b = b

    def prediction(self, Z : np.ndarray) -> np.ndarray:

        """Predict the output resulting from a given input

        Args:
            Z (ndarray of floats): The input on which to make the
                prediction. Must either be a one dimensional array of the
                same length as the number of calibration X variables, or a
                two dimensional array with the same number of columns as
                the calibration X data and one row for each input row.

        Returns:
            Y (ndarray of floats) : The predicted output - either a one
            dimensional array of the same length as the number of
            calibration Y variables or a two dimensional array with the
            same number of columns as the calibration Y data and one row
            for each input row.
        """

        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            return self.Y_offset + (Z - self.X_offset).T @ self.b.T
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            for i in range(0, Z.shape[0]):
                result[i, :] = self.Y_offset + \
                              (Z[i, :] - self.X_offset).T @ self.b.T
            return result

    def prediction_iterative(self, Z : np.ndarray) -> np.ndarray:

        """Predict the output resulting from a given input, iteratively

        This produces the same output as the one-step version ``prediction``
        but works by applying each loading in turn to extract the latent
        variables corresponding to the input.

        Args:
            Z (ndarray of floats): The input on which to make the
                prediction. Must either be a one dimensional array of the
                same length as the number of calibration X variables, or a
                two dimensional array with the same number of columns as
                the calibration X data and one row for each input row.

        Returns:
            Y (ndarray of floats) : The predicted output - either a one
            dimensional array of the same length as the number of
            calibration Y variables or a two dimensional array with the
            same number of columns as the calibration Y data and one row
            for each input row.
        """

        if len(Z.shape) == 1:
            if Z.shape[0] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            result[:, :] = self.Y_offset
            for k in range(0, self.Y_variables):
                x_j = Z - self.X_offset
                t = np.empty((self.components))
                for j in range(0, self.components):
                    t[j] = x_j @ self.W[k, :, j]
                    x_j = x_j - t[j] * self.P[k, :, j]
                result[k] += self.c[k, :] @ t

            return result

        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            result[:, :] = self.Y_offset
            for l in range(0, Z.shape[0]):
                for k in range(0, self.Y_variables):
                    x_j = Z[l, :] - self.X_offset
                    t = np.empty((self.components))
                    for j in range(0, self.components):
                        t[j] = x_j @ self.W[k, :, j]
                        x_j = x_j - t[j] * self.P[k, :, j]
                    result[l, k] += self.c[k, :] @ t

            return result
