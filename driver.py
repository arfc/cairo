import numpy as np
import pandas as pd
import sys, getopt

from tools import esn_prediction
from optimizers import grid_optimizer
from lorenz import generate_L96

# Optimization Sets
radius_set = [0.5, 0.7, 0.9,  1,  1.1,1.3,1.5]
noise_set = [ 0.0001, 0.0003,0.0007, 0.001, 0.003, 0.005, 0.007,0.01]
reservoir_set = [600, 800, 1000, 1500, 2000, 3000, 4000]
sparsity_set = [0.005, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
# This must change depending on the length of available data
trainingLengths = np.arange(4000,25000,300)

SOLAR_PATH = "./data/solarfarm_data.csv"
WIND_PATH = "./data/railsplitter_data.csv"
DEMAND_PATH = "./data/"

def main():
    pass


if __name__ == "__main__":
    inputfiles = []
    X_in = None
    options_dict = {'-u':'windspeed',
                    '-w':'wettemp',
                    '-d':'drytemp',
                    '-p':'pressure',
                    '-h':'humidity',
                    '-e':'solar elevation',
                    '-S':SOLAR_PATH,
                    '-D':DEMAND_PATH,
                    '-W':WIND_PATH,}

    # get arguments

    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   'uwdphei:f:o',
                                   ['infile=', 'altfile','outfile='])
    except getopt.GetoptError:
        print(f'Valid options are: {options_dict}')
        sys.exit(1)
    # print(args)
    for opt, arg in opts:
        print('option',opt)
        print(arg)

        if opt in ('-i', '--infile'):
            try:
                X_in = pd.read_csv(arg)
