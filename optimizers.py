import time
import numpy as np
import matplotlib.pyplot as plt
from tools import MSE, optimal_values, esn_prediction
from pyESN.pyESN import ESN


def grid_optimizer(
        data,
        params,
        args,
        xset,
        yset=None,
        verbose=False,
        visualize=False):
    """
    This function optimizes the ESN parameters, x and y, over a specified
    range of values. The optimal values are determined by minimizing
    the mean squared error. Those optimal values are returned.

    Parameters:
    -----------
    data : numpy array
        This is the dataset that the ESN should train and predict.
        If the training length plus the future total exceed the
        length of the data, an error will be thrown.
        **The shape of the transpose of the data will determine
        the number of inputs and outputs.**
    params : dictionary
        A dictionary containing all of the parameters required to
        initialize an ESN.
        Required parameters are:
            "n_reservoir" : int, the reservoir size
            "sparsity" : float, the sparsity of the reservoir
            "rand_seed" : int or None, specifies the initial seed
            "rho" : float, the spectral radius
            "noise" : the noise used for regularization
            "trainlen" : int, the training length
            "future" : int, the total prediction length
            "window" : int or None, the window size
    args : list or tuple
        The list of variables you want to optimize. Must be less
        than or equal to two.
    xset : numpy array
        The first set of values to be tested. Cannot be None.
    yset : numpy array or None
        The second set of values to be tested at the same
        time as the xset. Can be None.
    verbose : boolean
        Specifies if the simulation outputs should be printed.
        Useful for debugging.
    visualize : boolean
        Specifies if the results should be visualized.
    kwargs**:
        The keys of the ESN parameters that should be optimized.
        Correspond to xset and yset.

    Returns:
    --------
    loss : numpy array
        The array or matrix of loss values.
    """
    assert(len(args) <= 2), "Too many variables to optimize. Pick two or fewer."
    for variable in args:
        assert(variable in list(params.keys())
               ), f"{variable} not in parameters"

    if len(args) > 1:
        assert(yset is not None), "Two variables specified, two sets not given."

    xvar = args[0]
    loss = np.zeros(len(xset))

    if yset is not None:
        assert(len(args) > 1), "Second parameter set given, but not specified."
        yvar = args[1]
        loss = np.zeros([len(xset), len(yset)])

    if verbose:
        print(f"Optimizing over {args}:")

    predictLen = params['future']
    for x, xvalue in enumerate(xset):
        params[xvar] = xvalue
        if yset is not None:
            for y, yvalue in enumerate(yset):
                params[yvar] = yvalue
                predicted = esn_prediction(data, params)
                loss[x, y] = MSE(predicted, data[-predictLen:])

                if verbose:
                    print(
                        f"{xvar} = {xvalue}, {yvar} = {yvalue}, MSE={loss[x][y]}")

        else:
            predicted = esn_prediction(data, params)
            loss[x] = MSE(predicted, data[-predictLen:])

            if verbose:
                print(f"{xvar} = {xvalue}, MSE={loss[x]}")

    # =======================================================================
    # Visualization
    # =======================================================================

    if visualize and yset is not None:
        plt.figure(figsize=(16, 8))
        plt.title(f"Hyper-parameter Optimization over {args}")
        im = plt.imshow(loss.T,
                        vmin=abs(loss).min(),
                        vmax=abs(loss).max(),
                        origin='lower',
                        cmap='PuBu')
        plt.xticks(np.linspace(0, len(xset) - 1,
                               len(xset)),
                   xset)
        plt.yticks(np.linspace(0, len(yset) - 1,
                               len(yset)),
                   yset)
        plt.xlabel(f'{xvar}', fontsize=16)
        plt.ylabel(f'{yvar}', fontsize=16)
        cb = plt.colorbar(im)
        cb.set_label(label="Mean Squared Error",
                     fontsize=16,
                     rotation=-90,
                     labelpad=25)

    elif visualize and yset is None:
        plt.figure(figsize=(16, 9))
        plt.plot(xset, loss, '-ok', alpha=0.6)
        plt.title(f'MSE as a Function of {xvar}', fontsize=20)
        plt.xlabel(f'{xvar}', fontsize=18)
        plt.ylabel('MSE', fontsize=18)

    # =======================================================================
    # =======================================================================

    return loss
