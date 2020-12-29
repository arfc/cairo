import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import getopt
import os
import time

from tools import esn_prediction, optimal_values, param_string
from optimizers import grid_optimizer
from lorenz import generate_L63
from sunrise import generate_elevation_series

# Plot Parameters
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['figure.edgecolor'] = 'k'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"

# Optimization Sets
radius_set = [0.5, 0.7, 0.9, 1, 1.1,1.3,1.5]
noise_set = [ 0.0001, 0.0003,0.0007, 0.001, 0.003, 0.005, 0.007,0.01]

# radius_set = [0.1, 0.5, 1]
# noise_set = [0.001, 0.0007, 0.003]

reservoir_set = [600, 800, 1000, 1500, 2000, 3000, 4000]
sparsity_set = [0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]

# reservoir_set = [600, 800, 1000]
# sparsity_set = [0.005, 0.01, 0.2]

# This must change depending on the length of available data
trainingLengths = np.arange(4000, 25000, 300)

params = {'n_reservoir': 1000,
          'sparsity': 0.1,
          'rand_seed': 85,
          'rho': 1.5,
          'noise': 0.0001,
          'future': 500,
          'window': 500,
          'trainlen': 1000}

def main():
    pass


def get_variable_name(fname):
    """
    This function takes a file path and returns
    the name of a variable.
    """

    variables = ['demand', 'solarfarm', 'railsplitter']
    split_str = fname.split('/')
    file_name = split_str[-1]
    pieces = file_name.split('_')

    for p in pieces:
        if any(p in var for var in variables):
            return p
    return


if __name__ == "__main__":

    # =============================================================================
    # Set Up the Training Data
    # =============================================================================
    X_in = []
    t = None
    df = None
    wdf = None
    list_keys = None
    sun_elevation = None
    save_prefix = None


    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'LS:', ['save_prefix='])
    except getopt.GetoptError:
        print(f'Valid options are: {options_dict}')
        sys.exit(1)
    # print(args)
    for opt, arg in opts:
        if opt in ('-L'):
            t = np.arange(0,30,0.01)
            X_in = generate_L63(t)
        if opt in ('-S', '--save_prefix'):
            save_prefix = arg
# =============================================================================
# ESN Optimization
# =============================================================================
    MAX_TRAINLEN = int(len(X_in) - params['future'])
    print(f"Maximum training length is {MAX_TRAINLEN}")

    # pred = esn_prediction(X_in.T, params)
    print('Optimizing spectral radius and regularization')
    tic = time.perf_counter()
    radiusxnoise_loss = grid_optimizer(X_in,
                                       params,
                                       args=['rho', 'noise'],
                                       xset=radius_set,
                                       yset=noise_set,
                                       verbose=True,
                                       save_path=save_prefix)

    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"This simulation took {elapsed:0.02f} seconds")
    print(f"This simulation took {elapsed/60:0.02f} minutes")

    opt_radius, opt_noise = optimal_values(radiusxnoise_loss,
                                           radius_set,
                                           noise_set)
    params['rho'] = opt_radius
    params['noise'] = opt_noise

    print('Optimizing network size and sparsity')
    tic = time.perf_counter()
    sizexsparsity_loss = grid_optimizer(X_in,
                                        params,
                                        args=['n_reservoir', 'sparsity'],
                                        xset=reservoir_set,
                                        yset=sparsity_set,
                                        verbose=True,
                                        save_path=save_prefix)

    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"This simulation took {elapsed:0.02f} seconds")
    print(f"This simulation took {elapsed/60:0.02f} minutes")

    opt_size, opt_sparsity = optimal_values(sizexsparsity_loss,
                                            reservoir_set,
                                            sparsity_set)
    params['n_reservoir'] = opt_size
    params['sparsity'] = opt_sparsity

    trainingLengths = np.arange(100, MAX_TRAINLEN, 100)

    print('Optimizing training length')
    tic = time.perf_counter()
    trainlen_loss = grid_optimizer(X_in,
                                   params,
                                   args=['trainlen'],
                                   xset=trainingLengths,
                                   verbose=True,
                                   save_path=save_prefix)
    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"This simulation took {elapsed:0.02f} seconds")
    print(f"This simulation took {elapsed/60:0.02f} minutes")

    minloss = np.min(trainlen_loss)
    index_min = np.where(trainlen_loss == minloss)
    l_opt = trainingLengths[index_min][0]
    params['trainlen'] = l_opt

# =============================================================================
# ESN Prediction
# =============================================================================
    print("Generating optimized prediction...")
    tic = time.perf_counter()

    init_pred = esn_prediction(X_in, params, save_path=save_prefix)

    toc = time.perf_counter()
    elapsed = toc - tic
    print(f"This simulation took {elapsed:0.02f} seconds")
    print(f"This simulation took {elapsed/60:0.02f} minutes")


# =============================================================================
# Plot Prediction
# =============================================================================
    assert(save_prefix is not None), "No output filename given by user."
    target_folder = "./figures/"

    futureTotal = params['future']

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    plt.suptitle(f"Lorenz-63 Model Prediction with ESN", fontsize=21)
    plt.title(param_string(params))
    plt.figure(figsize=(16,9))
    futureTotal = params['future']
    ax1=plt.subplot(311)
    plt.plot(t[-2*futureTotal:], X_in[-2*futureTotal:, 0], label='Ground Truth')
    plt.plot(t[-futureTotal:], init_pred[:, 0], label='Prediction')
    ax2=plt.subplot(312, sharex=ax1)
    plt.plot(t[-2*futureTotal:], X_in[-2*futureTotal:, 1], label='Ground Truth')
    plt.plot(t[-futureTotal:], init_pred[:, 1], label='Prediction')
    ax3=plt.subplot(313, sharex=ax1)
    plt.plot(t[-2*futureTotal:], X_in[-2*futureTotal:, 2], label='Ground Truth')
    plt.plot(t[-futureTotal:], init_pred[:, 2], label='Prediction')



    ax1.set_ylabel("x")
    ax2.set_ylabel("y")
    ax3.set_ylabel("z")
    plt.legend()
    # save prefix should be something like "04_wind_elevation"
    # Check if there is a figures folder, if not, make one.
    plt.savefig(target_folder + save_prefix + '_prediction.png')
    plt.close()
