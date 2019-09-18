import pandas as pd
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt 

# import raw data -- the raw data is available for download on the box website.
path = "/home/dotson/Documents/cairo/willard_weather_data.csv"

weather_df = pd.read_csv(path, usecols=[1, 43, 44],) 
weather_df = weather_df.fillna(-999.99) # fill NaN values with -999+

# convert to datetime
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

# simplify column names
weather_df.rename(columns = {'HourlyDryBulbTemperature':'TEMP', 'HourlyPrecipitation':'PREC'}, inplace=True)

print(weather_df.head(5))


# convert temperatures to float, must use this method because there items like '47s'
for index, temp in enumerate(weather_df['TEMP']):
	if (type(temp) == str):
		string = temp
		temp_float = [float(i) for i in string.split() if i.isdigit()]
		if len(temp_float) > 0:
			weather_df.at[index,'TEMP'] = temp_float[0]
		else:
			weather_df.at[index,'TEMP'] = -999.99

# # converts all to float
weather_df['TEMP'] = weather_df['TEMP'].astype(float)

# # separates the date and time stamps, not sure if this is useful. 
# date, time = zip(*[(d.date(), d.time()) for d in weather_df['DATE']])
# weather_df = weather_df.assign(DATE=date, time=time)

# temp_df = weather_df[['DATE', 'TEMP', 'time']]
temp_df = weather_df[['DATE', 'TEMP']]
temp_df = temp_df[temp_df.TEMP != -999.99]
temp_df.plot(x='DATE', y='TEMP')
plt.show()


# ===================================================================
# Export cleaned datasets
# ===================================================================
dir_path = "/home/dotson/Documents/cairo/"
temp_df.to_csv(dir_path + "willard_temp.csv")


# ===================================================================
# Figure out which columns to use
# ===================================================================

# keys = weather_df.keys()
# i = 0
# print(keys[1])
# for key in keys:
# 	if ("Temperature" and "Hourly") in key:
# 		print(i, key)
	# i +=1
