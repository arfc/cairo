import numpy as np
from pyESN.pyESN import ESN


def MSE(yhat, y, ntargets=1):
    '''
    This function calculates the root mean squared error between
    a predicted and target vector.

    Parameters
    ----------
    yhat : numpy array
        The predicted, approximated, or calculated vector
    y : numpy array
        The target vector

    Returns
    -------
    mse : float
        The mean squared error between yhat and y.
    '''
    try:
        n_inputs = y.shape[1]
    except BaseException:
        n_inputs = 1

    if n_inputs > 1 and ntargets == 1:
        mse = np.sqrt(np.mean((y.T[0] - yhat.T[0])**2))
    else:
        mse = np.sqrt(np.mean((y.flatten() - yhat.flatten())**2))

    return mse


def NRMSE(yhat, y, ntargets=1):
    '''
    This function calculates the normalized root mean squared error
    between a predicted and target vector.

    Parameters
    ----------
    yhat : numpy array
        The predicted, approximated, or calculated vector
    y : numpy array
        The target vector

    Returns
    -------
    mse : float
        The mean squared error between yhat and y.
    '''

    mse = MSE(yhat, y, ntargets)
    try:
        n_inputs = y.shape[1]
    except BaseException:
        n_inputs = 1

    if n_inputs > 1 and ntargets == 1:
        sigma = np.std(y.T[0])
    else:
        sigma = np.std(y.flatten())
    nrmse = mse / sigma

    return nrmse


def MASE(yhat, y, training, ntargets=1, nsteps=1):
    '''
    This function calculates the mean absolute scaled
    error for a prediction.

    Parameters
    ----------
    yhat : numpy array
        The forecast vector
    y : numpy array
        The target vector
    ntargets : integer
        The number of target variables being predicted
    nsteps : integer
        The number of steps ahead for the forecast
    '''

    try:
        n_inputs = y.shape[1]
    except BaseException:
        n_inputs = 1

    if n_inputs > 1 and ntargets == 1:
        n = len(training.T[0])
        et = y.T[0] - yhat.T[0]
        # print(training.T.shape)
        # print(training.T[0, nsteps:].shape)
        rdwalk = np.sum(
            np.abs(training.T[0, nsteps:] - training.T[0, :-nsteps]))
        qt = (n - nsteps) * (et) / (rdwalk)
        mase = np.mean(abs(qt))
    else:
        # print("Flattening...")
        y = y.flatten()
        yhat = yhat.flatten()
        n = len(training)
        et = y - yhat
        print(f'training shape {training.shape}')
        rdwalk = np.sum(
            np.abs(
                training.flatten()[
                    nsteps:] -
                training.flatten()[
                    :-
                    nsteps]))
        qt = (n - nsteps) * (et) / (rdwalk)
        mase = np.mean(abs(qt))

    return mase


def MAE(yhat, y, ntargets=1):
    '''
    This function calculates the mean absolute error between
    a predicted and target vector.

    Parameters
    ----------
    yhat : numpy array
        The predicted, approximated, or calculated vector
    y : numpy array
        The target vector

    Returns
    -------
    mae : float
        The mean squared error between yhat and y.
    '''
    try:
        n_inputs = y.shape[1]
    except BaseException:
        n_inputs = 1

    if n_inputs > 1 and ntargets == 1:
        mae = np.mean(np.abs(y.T[0] - yhat.T[0]))
    else:
        mae = np.mean(np.abs(y.flatten() - yhat.flatten()))

    return mae


def param_string(params):
    """
    This function generates a formatted string from
    model parameters.

    Parameters
    ----------
    params : dictionary
        A dictionary containing all of the parameters required to
        initialize an ESN.

        Required parameters are:
            * "n_reservoir" : int, the reservoir size
            * "sparsity" : float, the sparsity of the reservoir
            * "rand_seed" : int or None, specifies the initial seed
            * "rho" : float, the spectral radius
            * "noise" : the noise used for regularization
            * "trainlen" : int, the training length
            * "future" : int, the total prediction length
            * "window" : int or None, the window size

    Returns
    -------
    pstring : string
        The formatted parameter string.
    """

    n_reservoir = params['n_reservoir']
    sparsity = params['sparsity']
    spectral_radius = params['rho']
    noise = params['noise']
    trainlen = params['trainlen']
    window = params['window']

    pstring = (f"Reservoir Size:{n_reservoir}, Sparsity: {sparsity}, "
               f"Spectral Radius: {spectral_radius}, Noise: {noise}, "
               f"Training Length: {trainlen}, "
               f"Prediction Window: {window}")

    return pstring


def optimal_values(loss, xset, yset):
    """
    This function returns the optimal set of values given
    a matrix of error values. The optimal set is the pair
    of values that minimizes the error.

    Parameters
    ----------
    loss : numpy matrix
        The matrix of loss values.
    xset : numpy matirx
    yset: numpy matirx

    Returns
    -------
    x, y : float
        The optimal set of values
    """

    minLoss = np.min(loss)
    index_min = np.where(loss == minLoss)
    x_optimal = xset[int(index_min[0])]
    y_optimal = yset[int(index_min[1])]

    return x_optimal, y_optimal


def esn_prediction(data, params, save_path=None):
    """
    This function generates a prediction with an ESN over
    the specified time range. Currently, only n_inputs=n_outputs
    is supported.

    Parameters
    ----------
    data : numpy array
        This is the dataset that the ESN should train and predict.
        If the training length plus the future total exceed the
        length of the data, an error will be thrown.
        **The shape of the transpose of the data will determine
        the number of inputs and outputs.**

            E.g. Two datasets trained together

            >>> X_in = np.concatenate([[set1, set2]], axis=1)
            >>> pred = esn_prediction(X_in.T, params)

    params : dictionary
        A dictionary containing all of the parameters required to
        initialize an ESN.

        Required parameters are:
            * "n_reservoir" : int, the reservoir size
            * "sparsity" : float, the sparsity of the reservoir
            * "rand_seed" : int or None, specifies the initial seed
            * "rho" : float, the spectral radius
            * "noise" : the noise used for regularization
            * "trainlen" : int, the training length
            * "future" : int, the total prediction length
            * "window" : int or None, the window size

    save_path : string
        Save the prediction data to this location as a .npy file.

    Return
    ------
    prediction : numpy array
        The prediction generated by the ESN. Should have the
        same second dimension as data.
    """
    trainlen = params['trainlen']
    window = params['window']
    futureTotal = params['future']

    if window is not None:
        assert(futureTotal % window == 0), "Window must be multiple of future."

    # get the shape
    ndims = len(data.shape)
    if ndims > 1:
        n_vars = data.shape[1]
    else:
        n_vars = 1

    esn = ESN(n_inputs=n_vars,
              n_outputs=n_vars,
              n_reservoir=params['n_reservoir'],
              sparsity=params['sparsity'],
              random_state=params['rand_seed'],
              spectral_radius=params['rho'],
              noise=params['noise'])

    # train the ESN
    prediction = np.ones((futureTotal, n_vars))
    window_pred = np.ones((window, n_vars))

    for i in range(0, futureTotal, window):
        data_slice = data[-trainlen - futureTotal + i:-futureTotal + i]
        pred_training = esn.fit(np.ones((trainlen, n_vars)),
                                data_slice)
        inter_pred = esn.predict(window_pred)
        prediction[i:i + window] = inter_pred

    # ===================================================
    # Save Data
    # ===================================================
    if save_path is not None:
        np.save("./data/" + save_path + "_prediction", prediction)
        np.save("./data/" + save_path + "_input", data)

    return prediction


def esn_scenario(data, params):
    """
    This function generates a prediction with an ESN over
    the specified time range. Currently, only n_inputs=n_outputs
    is supported.

    Parameters
    ----------
    data : numpy array
        This is the dataset that the ESN should train and predict.
        If the training length plus the future total exceed the
        length of the data, an error will be thrown.
        **The shape of the transpose of the data will determine
        the number of inputs and outputs.**

            E.g. Two datasets trained together

            >>> X_in = np.concatenate([[set1, set2]], axis=1)
            >>> pred = esn_prediction(X_in.T, params)

    params : dictionary
        A dictionary containing all of the parameters required to
        initialize an ESN.

        Required parameters are:
            * "n_reservoir" : int, the reservoir size
            * "sparsity" : float, the sparsity of the reservoir
            * "rand_seed" : int or None, specifies the initial seed
            * "rho" : float, the spectral radius
            * "noise" : the noise used for regularization
            * "trainlen" : int, the training length
            * "future" : int, the total prediction length
            * "window" : int or None, the window size

    Return
    ------
    prediction : numpy array
        The prediction generated by the ESN. Should have the
        same second dimension as data.
    """

    # get the shape
    ndims = len(data.shape)
    if ndims > 1:
        n_vars = data.shape[1]
    else:
        n_vars = 1

    esn = ESN(n_inputs=n_vars,
              n_outputs=n_vars,
              n_reservoir=params['n_reservoir'],
              sparsity=params['sparsity'],
              random_state=params['rand_seed'],
              spectral_radius=params['rho'],
              noise=params['noise'])

#    trainlen = params['trainlen']
    trainlen = len(data)
    futureTotal = params['future']
    pred_tot = np.ones((futureTotal, n_vars))

    # train the ESN
    pred_training = esn.fit(np.ones((trainlen, n_vars)),
                            data)
    scenario = esn.predict(pred_tot)

    return scenario, esn
