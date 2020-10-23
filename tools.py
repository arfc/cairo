import numpy as np


def MSE(yhat, y):
    '''
    This function calculates the mean squared error between
    a predicted and target vector.

    Parameters:
    -----------
    yhat : numpy array
        The predicted, approximated, or calculated vector
    y : numpy array
        The target vector

    Returns:
    --------
    mse : float
        The mean squared error between yhat and y.
    '''
    mse = np.sqrt(np.mean((yhat.flatten() - y)**2))

    return mse


def optimal_values(loss, xset, yset):
    """
    This function returns the optimal set of values given
    a matrix of error values. The optimal set is the pair
    of values that minimizes the error.

    Parameters:
    -----------
    loss : numpy matrix
        The matrix of loss values.

    Returns:
    --------
    x, y : float
        The optimal set of values
    """

    minLoss = np.min(loss)
    index_min = np.where(loss == minLoss)
    x_optimal = xset[int(index_min[0])]
    y_optimal = yset[int(index_min[1])]

    return
