from sunrise import generate_elevation_series
from optimizers import grid_optimizer
from tools import MSE, MAE, NRMSE, MASE
from tools import esn_prediction, optimal_values, param_string
import time
import os
import getopt
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("pgf")


# Plot Parameters
# plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.edgecolor'] = 'k'
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['pgf.rcfonts'] = False

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
trainingLengths = np.arange(5000, 25000, 300)

params = {'n_reservoir': 1000,
          'sparsity': 0.1,
          'rand_seed': 85,
          'rho': 1.5,
          'noise': 0.0001,
          'future': 48,
          'window': 48,
          'trainlen': 5000}

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

# get the command line arguments

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
            # params['future'] = int(arg)

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
    # uncomment for 2 norm
    # power_norm = np.linalg.norm(power)

    # uncomment for infinity norm
    power_norm = np.linalg.norm(power, ord=np.inf)
    data_norms.append(power_norm)
    X_in.append(power / power_norm)

    if sun_elevation is not None:
        # uncomment for 2 norm
        # elevation_norm = np.linalg.norm(sun_elevation)

        # uncomment for infinity norm
        elevation_norm = np.linalg.norm(sun_elevation, ord=np.inf)
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
                # uncomment for 2 norm
                # aspect_norm = np.linalg.norm(aspect_data)

                # uncomment for infinity norm
                aspect_norm = np.linalg.norm(aspect_data, ord=np.inf)
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
    plt.title(f'RMSE as a Function of Training Length')
    plt.xlabel(f'Training Length')
    plt.ylabel('MSE')
    plt.savefig("./images/" + save_prefix + "_trainlen_loss.pgf")
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
    nrmse = NRMSE(init_pred, X_in.T[-futureTotal:])

    startTrain = params['trainlen']
    endTrain = futureTotal
    trainset = X_in.T[-startTrain:-endTrain]
    mase = MASE(init_pred, X_in.T[-futureTotal:],
                ntargets=1,
                training=trainset,
                nsteps=params['window'])
# =============================================================================
# Plot Prediction
# =============================================================================
    assert(save_prefix is not None), "No output filename given by user."
    target_folder = "./images/"

    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    var = get_variable_name(datafile_name)
    colwidth = 3.07242 * 2
    height = 0.5 * colwidth
    plt.figure(figsize=(colwidth, height))
    plt.title(f"{VARIABLES[var]} Prediction with an ESN")
    # plt.title(param_string(params))
    plt.ylabel("Energy [kWh]")
    plt.xlabel(f"Hours since {df.index[0]}")
    # plot the truth
    hours = np.arange(0, len(xdf.index), 1)
    plt.plot(hours[-2 * futureTotal:], xdf.kw[-2 * futureTotal:],
             'b', label=f"True {VARIABLES[var]}",
             alpha=0.7,
             color='tab:blue')
    # # plot the prediction
    plt.plot(hours[-futureTotal:], power_norm * init_pred.T[0], alpha=0.8,
             label='ESN Prediction',
             color='tab:red',
             linestyle='-')
    plt.legend(loc='upper left')
    if any(init_pred.T[0] < 0):
        x = hours[-futureTotal:]
        y1 = 0
        y2 = power_norm * init_pred.T[0]
        plt.axhline(y=y1)
        plt.fill_between(x, y1, y2,
                         where=(y2 <= y1),
                         linestyle='-',
                         color='gray',
                         alpha=0.6)

    # save prefix should be something like "04_wind_elevation"
    plt.savefig(target_folder + save_prefix + '_prediction.pgf')
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
        file.write(f"Normalized RMSE: {nrmse}\n")
        file.write(f"Mean Absolute Scaled Error: {mase}\n")
        file.write("\n")
