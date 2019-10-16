"""
This file reads all data from the AlsoEnergy website and
concatenates it into a single dataframe.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# To import from box directory
# from boxsdk import Client
# from boxsdk import OAuth2

# To import from local directory
import os


if __name__ == '__main__':

    # Import from local directory
    # Path to solar panel data
    path = "/home/dotson/Documents/cairo/solar_data/"

    # Gets file names
    files = [file for file in os.listdir(path) if file.endswith(".csv")]

    # # Import from box.com --> Not working.
    # oauth = OAuth2(client_id='sgd2, client_secret='SECRET',
    # access_token='TOKEN')
    # client = Client(oauth)

    # user = client.user().get()
    # print('The current user ID is {0}'.format(user.id))

    # shared_folder = client.get_shared_item(
    # "https://uofi.box.com/s/46lfkbv2lij1l9qrt52qif6fv9o0tlq2")
    # for file in shared_folder.get_items():
    # 	if item.type == 'file':
    # 		print(file)

    keylist = ['New Nexus 1272 Meter', 'Inverters']

    frames = []
    for file in files:
        # print(file)
        # Read in csv file as dataframe
        df = pd.read_csv(
            path + file,
            skiprows=[0, 2],
            index_col='Timestamp',
            parse_dates=True,
            dtype={
                keylist[0]: np.float64,
                keylist[1]: np.float64})
        # Drop NaN values
        df = df.dropna()
        # Add to list of dataframes
        # frames.append(df)
        if list(df.keys()) == keylist:
            frames.append(df)
        else:
            pass

    df_total = pd.concat(frames)

    df_total.plot()
    plt.show()
