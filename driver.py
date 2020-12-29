import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import getopt
import os
import time

from tools import esn_prediction, optimal_values, param_string, MSE, MAE
from optimizers import grid_optimizer
from lorenz import generate_L96
from sunrise import generate_elevation_series

# Plot Parameters
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.edgecolor'] = 'k'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"

# Optimization Sets
radius_set = [0.5, 0.7, 0.9, 1, 1.1, 1.2, 1.3, 1.5]
noise_set = [0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.005, 0.007, 0.01]

# radius_set = [0.1, 0.5, 1]
# noise_set = [0.001, 0.0007, 0.003]

reservoir_set = [600, 800, 1000, 1500, 2000, 2500, 3000, 4000]
sparsity_set = [0.005, 0.01, 0.03, 0.05, 0.1, 0.12, 0.15, 0.2]

# reservoir_set = [600, 800, 1000]
# sparsity_set = [0.005, 0.01, 0.2]

# This must change depending on the length of available data
trainingLengths = np.arange(4000, 25000, 300)

params = {'n_reservoir': 1000,
          'sparsity': 0.1,
          'rand_seed': 85,
          'rho': 1.5,
          'noise': 0.0001,
          'future': 96,
          'window': 96,
          'trainlen': 8000}

VARIABLES = {
    'solarfarm': 'Solar Generation',
    'railsplitter': 'Wind Generation',
    'demand': 'Demand'}


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
    data_norms = []
    datafile_name = None
    df = None
    wdf = None
    list_keys = None
    sun_elevation = None
    save_prefix = None
    options_dict = {'-u': 'windspeed',
                    '-w': 'wettemp',
                    '-d': 'drytemp',
                    '-p': 'pressure',
                    '-h': 'humidity',
                    }

    # get arguments

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'uwdpheH:i:f:oS:',
                                   ['infile=', 'altfile', 'outfile=',
                                    'save_prefix='])
    except getopt.GetoptError:
        print(f'Valid options are: {options_dict}')
        sys.exit(1)
    for opt, arg in opts:
        if opt in ('-i', '--infile'):
            try:
                df = pd.read_csv(arg,
                                 usecols=['time', 'kw'],
                                 index_col='time',
                                 parse_dates=True)
            except FileNotFoundError:
                print(f"Data file {arg} not found")

            datafile_name = arg
        if opt in ('-f', '--altfile'):
            assert (df is not None), "No data to predict"
            try:
                wdf = pd.read_csv(arg,
                                  index_col='time',
                                  parse_dates=True)
            except FileNotFoundError:
                print(f"Data file {arg} not found")

            list_keys = []
            for key in options_dict.keys():
                if any(key in option for option in opts):
                    list_keys.append(options_dict[key])
                    print(f"listkeys: {list_keys}")

        if opt in ('-e'):
            assert (df is not None), "No data to predict"
            sun_elevation = generate_elevation_series(
                df.index, timestamps=True)

        if opt in ('-S', '--save_prefix'):
            save_prefix = arg

        if opt in ('-H'):
            params['window'] = int(arg)

    # Align the two dataframes
    if wdf is not None:
        print("joining dataframes")
        xdf = pd.concat([df, wdf], axis=1, join='inner')
        xdf.interpolate('linear', inplace=True)
        print(xdf.head())
    else:
        xdf = df

    # Get the training data
    power = np.array(xdf.kw).astype('float64')
    power_norm = np.linalg.norm(power)
    data_norms.append(power_norm)
    X_in.append(power / power_norm)

    if sun_elevation is not None:
        elevation_norm = np.linalg.norm(sun_elevation)
        data_norms.append(elevation_norm)
        X_in.append(sun_elevation / elevation_norm)

    if list_keys is not None:
        for key in list_keys:
            if key == '-e':
                pass
            else:
                print(f"Adding key {key}")
                # "Aspect" refers to data for a particular aspect of "weather"
                aspect_data = np.array(xdf[key]).astype('float64')
                aspect_norm = np.linalg.norm(aspect_data)
                data_norms.append(aspect_norm)
                X_in.append(aspect_data / aspect_norm)

    X_in = np.array(X_in)
    print(X_in.shape, len(X_in.shape))


# =============================================================================
# ESN Optimization
# =============================================================================
    MAX_TRAINLEN = int(len(xdf) - params['future'])
    print(f"Maximum training length is {MAX_TRAINLEN}")

    print('Optimizing spectral radius and regularization')
    tic = time.perf_counter()
    radiusxnoise_loss = grid_optimizer(X_in.T,
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
    sizexsparsity_loss = grid_optimizer(X_in.T,
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

    trainingLengths = np.arange(5000, MAX_TRAINLEN, 300)

    print('Optimizing training length')
    tic = time.perf_counter()
    trainlen_loss = grid_optimizer(X_in.T,
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
# Visualize Training Length Loss
# =============================================================================

    plt.plot(trainingLengths, trainlen_loss, '-ok', alpha=0.6)
    plt.title(f'MSE as a Function of Training Length', fontsize=20)
    plt.xlabel(f'Training Length', fontsize=18)
    plt.ylabel('MSE', fontsize=18)
    plt.savefig("./figures/" + save_prefix + "_trainlen_loss.png")
    plt.close()
# =============================================================================
# ESN Prediction
# =============================================================================
    print("Generating optimized prediction...")
    tic = time.perf_counter()

    init_pred = esn_prediction(X_in.T, params, save_path=save_prefix)

    toc = time.perf_counter()
    elapsed = toc - tic
    prediction_time = elapsed
    print(f"This simulation took {elapsed:0.02f} seconds")
    print(f"This simulation took {elapsed/60:0.02f} minutes")

    futureTotal = params['future']

    rmse = MSE(init_pred, X_in.T[-futureTotal:])
    mae = MAE(init_pred, X_in.T[-futureTotal:])


# =============================================================================
# Plot Prediction
# =============================================================================
    assert(save_prefix is not None), "No output filename given by user."
    target_folder = "./figures/"

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    var = get_variable_name(datafile_name)

    plt.suptitle(f"{VARIABLES[var]} Prediction with ESN", fontsize=21)
    plt.title(param_string(params))
    plt.ylabel("Energy [kWh]", fontsize=16)
    plt.xlabel(f"Hours since {df.index[0]}", fontsize=16)
    # plot the truth
    plt.plot(xdf.index[-2 * futureTotal:], xdf.kw[-2 * futureTotal:],
             'b', label=f"True {VARIABLES[var]}",
             alpha=0.7,
             color='tab:blue')
    # # plot the prediction
    plt.plot(xdf.index[-futureTotal:], power_norm * init_pred.T[0], alpha=0.8,
             label='ESN Prediction',
             color='tab:red',
             linestyle='-')
    plt.legend()

    # save prefix should be something like "04_wind_elevation"
    plt.savefig(target_folder + save_prefix + '_prediction.png')
    plt.close()

# =============================================================================
# Save Metadata
# =============================================================================

    with open('./data/simulation_MD.txt', 'a') as file:
        file.write("==========================================\n")
        file.write(f"Metadata for [{save_prefix}]\n")
        file.write(f"{param_string(params)}\n")
        file.write(f"Random state: {params['rand_seed']}\n")
        file.write(f"Optimized prediction took: {prediction_time} seconds\n")
        file.write(f"Mean Absolute Error: {mae}\n")
        file.write(f"Root Mean Squared Error: {rmse}\n")
        file.write("\n")
