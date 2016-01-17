"""A module which implements the PLS-SB algorithm."""

from . import *


class PLS_SB(RegressionBase):

    """Regression using the PLS-SB algorithm.

    The PLS-SB sets up the same mathematical problem as the PLS2 module,
    but then formulates the convergence criteria as an eigenvalue problem
    and solves it directly. It is therefore a deterministic algorithm, but
    has the drawback that all components must be extracted at once, even
    if only a few are required. Note that the output of PLS-SB is not the
    same as PLS2. In the PLS2 each component found is removed from the
    working copies of the input matrices by a rank-1 operation so the next
    iterations will converge on a new component. In PLS-SB all components
    are found at once.

    Args:
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract

    Note:
        The attributes of the resulting class are exactly the same as for
        :py:class:`pls2.PLS2`.

    """

    def __init__(self, X, Y, g):

        Xc, Yc = super()._prepare_data(X, Y)

        if g < 1 or g > self.max_rank:
            raise ParameterError('Number of required components '
                                 'specified is impossible.')
        self.components = g

        XtY = Xc.T @ Yc
        _, W = linalg.eigh(XtY @ XtY.T)

        self.W = W[:, :-g-1:-1].real

        self.T = Xc @ self.W
        self.Q = Yc.T @ self.T
        self.Q /= np.linalg.norm(self.Q, axis=0)
        self.U = Yc @ self.Q
        t_dot_t = (self.T.T @ self.T).diagonal()
        self.C = np.diag((self.T.T @ self.U).diagonal() / t_dot_t)
        self.P = (Xc.T @ self.T) / t_dot_t

        self.B = self.W @ linalg.inv(self.P.T @ self.W) @ self.C @ self.Q.T

    def prediction(self, Z):

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
            return self.Y_offset + (Z - self.X_offset).T @ self.B
        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            for i in range(0, Z.shape[0]):
                result[i, :] = self.Y_offset + \
                              (Z[i, :] - self.X_offset).T @ self.B
            return result

    def prediction_iterative(self, Z):

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

            x_j = Z - self.X_offset
            t = np.empty((self.components))
            for j in range(0, self.components):
                t[j] = x_j @ self.W[:, j]
                x_j = x_j - t[j] * self.P[:, j]
            result = self.Y_offset + t  @ self.C @ self.Q.T

            return result

        else:
            if Z.shape[1] != self.X_variables:
                raise ParameterError('Data provided does not have the  same '
                                     'number of variables as the original X '
                                     'data')
            result = np.empty((Z.shape[0], self.Y_variables))
            t = np.empty((self.components))

            for k in range(0, Z.shape[0]):
                x_j = Z[k, :] - self.X_offset
                for j in range(0, self.components):
                    t[j] = x_j @ self.W[:, j]
                    x_j = x_j - t[j] * self.P[:, j]
                result[k, :] = self.Y_offset + t  @ self.C @ self.Q.T

            return result
