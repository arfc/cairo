import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# import weather data
path = "/home/dotson/Documents/cairo/willard_weather_data.csv"

raw_data = pd.read_csv(path, parse_dates=['DATE'], index_col=['DATE'], usecols=[1, 43, 44])
raw_data.plot()

# Figure out which columns to use

# keys = raw_data.keys()
# i = 0
# print(keys[1])
# for key in keys:
# 	if ("Temperature" and "Hourly") in key:
# 		print(i, key)
	# i +=1

print(raw_data.head(30))
# fill NaN values with -999
# raw_data.iloc[:,0] = pd.to_datetime(raw_data.iloc[:,0])
raw_data = raw_data.fillna(-999.99)

x = raw_data.iloc[:,0]
y = raw_data.iloc[:,1]
# plt.plot(x,y)
# plt.show()
# raw_data.plot(x=DATE, y=HourlyDryBulbTemperature, kind="line")

print(raw_data.head(30))



