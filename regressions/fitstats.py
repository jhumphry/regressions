"""A module which implements goodness-of-fit statistics."""

try:
    import scipy.stats
    _stats_available = True
except ImportError:
    _stats_available = False

from . import *


def SS(Y):
    """Implements the Sum of Squares

    This function calculates the sum of the squared input data. The input
    data is first centered by subtracting the mean.

    Args:
        Y (ndarray N x m): Y calibration data, one row per data sample

    Returns:
        SS (float): The sum of the squares of the input data.

    """

    # Change 1-D array into column vector
    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))

    Yc = Y - Y.mean(0)

    return (Yc**2.0).sum()


def RESS(R, X, Y, others=None, relative=False):
    """Implements the Residual Error Sum of Squares

    This function calculates the RESS statistic for a given regression
    class and a set of calibration data. The regression function is
    trained on the X and Y data. The X data is then used to predict a set
    of Y data. The difference between these predictions and the true Y
    data is squared and summed to give the RESS statistic. Note that this
    statistic can be misleading if used on its own as it can reward
    routines which over-fit to the sample data and do not have good
    generalisation performance. Consider using in conjunction with the
    :py:func:`PRESS` statistic.

    Args:
        R (class): A regression class
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        others (dict, optional): A dict of other parameters to send to the
            regression class constructor.
        relative (boolean, optional): whether to divide the error by the
            true Y value before squaring and summing - where Y columns have
            different scales this may help to prevent the output being
            dominated by the column with the largest magnitude.

    Returns:
        RESS (float): The RESS statistic.

    """

    if others is None:
        others = {}

    if X.shape[0] != Y.shape[0]:
        raise ParameterError('X and Y data must have the same number of '
                             'rows (data samples)')

    # Change 1-D arrays into column vectors
    if len(X.shape) == 1:
        X = X.reshape((X.shape[0], 1))

    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))

    model = R(X=X, Y=Y, **others)
    Yhat = model.prediction(Z=X)

    if relative:
        return (((Yhat - Y) / Y)**2).sum()
    else:
        return ((Yhat - Y)**2).sum()


def R2(R, X, Y, others=None):
    """Implements the R**2 statistic

    This function calculates the R**2 statistic for a given regression
    class and a set of calibration data. This is equal to (1-RESS/SS),
    which gives an indication of how much of the initial variation in the
    (centered) Y data is explained by the regression model after it has
    been trained on the same Y data. Note that an overfitted model can
    have a very large R**2 but poor generalisation performance. The
    :py:func:`Q2` statistic looks at how much variance in each part of the
    Y data is explained by the regression model trained on only the other
    parts of the Y data so is more robust against overfitting.

    Args:
        R (class): A regression class
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        others (dict, optional): A dict of other parameters to send to the
            regression class constructor.

    Returns:
        R2 (float): The R2 statistic.

    """

    return 1.0 - RESS(R, X, Y, others, relative=False) / SS(Y)


def PRESS(R, X, Y, groups=4, others=None, relative=False):
    """Implements the Predicted Residual Error Sum of Squares

    This function calculates the PRESS statistic for a given regression
    class and a set of calibration data. Each groups of samples in turn is
    removed from the data set, the regression model is trained on the
    remaining data, and then is used to predict the Y values of the
    samples that were removed. Once a full set of Y predictions has been
    produced, the sum of the squared difference between them and the true
    Y data is the PRESS statistic.

    Args:
        R (class): A regression class
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        groups (int, optional): Number of cross-validation groups to use
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

    if data_samples < groups:
        raise ParameterError('There must be at least as many data samples as '
                             'cross-validation groups')

    if groups < 2:
        raise ParameterError('There must be at least two cross-validation '
                             'groups')

    group_size = data_samples // groups
    start_indexes = [x*group_size for x in range(0, groups)]
    end_indexes = [x*group_size for x in range(1, groups+1)]
    end_indexes[-1] = data_samples  # Last group may be bigger

    # Change 1-D arrays into column vectors
    if len(X.shape) == 1:
        X = X.reshape((X.shape[0], 1))

    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))

    Yhat = np.empty(Y.shape)

    for i in range(0, groups):

        samples_excluding_group = data_samples - \
            (end_indexes[i]-start_indexes[i])
        Xp = np.empty((samples_excluding_group, X.shape[1]))
        Yp = np.empty((samples_excluding_group, Y.shape[1]))

        Xp[0:start_indexes[i], :] = X[0:start_indexes[i], :]
        Xp[start_indexes[i]:, :] = X[end_indexes[i]:, :]
        Yp[0:start_indexes[i], :] = Y[0:start_indexes[i], :]
        Yp[start_indexes[i]:, :] = Y[end_indexes[i]:, :]

        model = R(X=Xp, Y=Yp, **others)
        Yhat[start_indexes[i]:end_indexes[i], :] = \
            model.prediction(Z=X[start_indexes[i]:end_indexes[i], :])

    if relative:
        return (((Yhat - Y) / Y)**2).sum()
    else:
        return ((Yhat - Y)**2).sum()


def Q2(R, X, Y, groups=4, others=None):
    """Implements the Q**2 statistic

    This function calculates the Q**2 statistic for a given regression
    class and a set of calibration data. This is equal to (1-PRESS/SS),
    which gives an indication of how much of the initial variation in each
    part of the (centered) Y data is explained by the regression model
    trained on the other parts of the Y data. This attempts to ensure that
    regression models with a tendency to over-fit training data are not
    favoured.

    Args:
        R (class): A regression class
        X (ndarray N x n): X calibration data, one row per data sample
        Y (ndarray N x m): Y calibration data, one row per data sample
        groups (int, optional): Number of cross-validation groups to use
        others (dict, optional): A dict of other parameters to send to the
            regression class constructor.

    Returns:
        Q2 (float): The Q2 statistic.

    """

    return 1.0 - PRESS(R, X, Y, groups, others, relative=False) / SS(Y)


def residuals_QQ(Y):
    """Function for creating normal Q-Q probability plots of residuals

    This function is used to explore the residuals left over after a
    regression model has been fitted to some calibration data. The input
    is a matrix of residuals created by subtracting the true Y calibration
    values from the Y values predicted by the regression model when the X
    calibration values are input. Each column represents a variable of Y,
    and in turn each is centered, divided by the standard deviation of the
    values in the column and sorted.

    Theoretical quantiles from the normal distribution and the sample
    quantiles for each Y variable are returned. When the theoretical
    quantiles are plotted against the sample quantiles for any of the Y
    variables, a Q-Q plot is producted. If the residuals are normally
    distributed, the points should lie on a straight line through the
    origin.

    Requires 'SciPy' to be available.

    Args:
        Y (ndarray N x m): Matrix of residuals

    Returns:
        X, Y (tuple of ndarray N and ndarray N x m): The theoretical
        quantiles from the normal distribution and the sample quantiles
        from the normal distribution

    Raises:
        NotImplementedError: SciPy is not available

    """

    if not _stats_available:
        raise NotImplementedError("This function requires SciPy")

    # Change 1-D array into column vector
    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))

    Yc = (Y - Y.mean(0))
    Yc /= Yc.std(0)
    Yc.sort(0)

    samples = Y.shape[0]
    X = np.empty((samples))
    for i in range(0, samples):
        X[i] = scipy.stats.norm.ppf(1.0 / (samples+1) * (i+1))

    return X, Yc
