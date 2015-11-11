"""A module which implements PRESS."""

from . import *


def PRESS(R, X, Y, g, others=None, relative=False):
    """Implements the Predicted Residual Error Sum of Squares

    This function calculates the PRESS statistic for a given regression
    class and a set of calibration data. Each sample in turn is removed
    from the data, the regression model is trained on the remaining data
    and then used to predict the Y value of the sample that was removed.
    Once a full set of Y predictions has been produced, the sum of the
    squared difference between them and the true Y data is the PRESS
    statistic.

    Args:
        R (class): A regression class
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        g (int): Number of components to extract
        others (dict, optional): A dict of other parameters to send to the
            regression class constructor.
        relative (boolean, optional): whether to divide the error by the
            true Y value before squaring and summing - where Y columns have
            different scales this may help to prevent the output being
            dominated by the column with the largest magnitude.

    Returns:
        PRESS (float): The PRESS statistic.

    """

    if others is None:
        others = {}

    if X.shape[0] != Y.shape[0]:
        raise ParameterError('X and Y data must have the same number of '
                             'rows (data samples)')

    data_samples = X.shape[0]

    if data_samples < 2:
        raise ParameterError('There must be at least two data samples to '
                             'produce the PRESS statistic.')

    Xp = np.empty((X.shape[0] - 1, X.shape[1]))
    Yp = np.empty((Y.shape[0] - 1, Y.shape[1]))
    Yhat = np.empty(Y.shape)

    for i in range(0, data_samples):
        Xp[0:i, :] = X[0:i, :]
        Xp[i:, :] = X[i+1:, :]
        Yp[0:i, :] = Y[0:i, :]
        Yp[i:, :] = Y[i+1:, :]
        model = R(X=Xp, Y=Yp, g=g, **others)
        Yhat[i, :] = model.prediction(Z=X[i, :])

    if relative:
        return (((Yhat - Y) / Y)**2).sum()
    else:
        return ((Yhat - Y)**2).sum()
