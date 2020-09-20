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
