import pandas as pd
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt 

# ===================================================================
# Useful functions
# ===================================================================

def isfloat(value):
	"""
	This function checks if a string can be converted to a float.

	Parameters:
	-----------
	value : string

	Returns:
	--------
	boolean
	"""
	try:
		float(value)
		return True
	except ValueError:
		return False


def to_float(dataframe, column):
	"""
	This function converts all values in a specified column to float.
	It is necessary to create my own function because the data has some
	errors in it that prevent it from being parsed as a single data type.

	Parameters:
	-----------
	dataframe : pandas dataframe
	column : string
		The name of the column in the dataframe that
	
	Returns:
	--------
	float_df : pandas dataframe
		This is the correctly formatted dataframe with the correct dtype.
	"""

	float_df = dataframe

	for index, value in enumerate(float_df[column]):

		if isfloat(value):
			float_df.at[index,column] = float(value)
		elif (type(value) == str):
			value_float = [float(i) for i in value.split() if i.isdigit()]
			if len(value_float) > 0:
				# If there was a value at the location
				float_df.at[index,column] = value_float[0]
			else:
				# If the value at location was NaN
				float_df.at[index,column] = -999.99

	float_df[column] = float_df[column].astype(float)
	return float_df


if __name__ == '__main__':

	# import raw data -- the raw data is available for download on the box website.
	path = "/home/dotson/Documents/cairo/willard_weather_data.csv"
	
	weather_df = pd.read_csv(path, usecols=[1, 43, 44],) 
	weather_df = weather_df.fillna(-999.99) # fill NaN values with -999+

	# convert to datetime
	weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

	# simplify column names
	weather_df.rename(columns = {'HourlyDryBulbTemperature':'TEMP', 'HourlyPrecipitation':'PREC'}, inplace=True)

	# # separates the date and time stamps, not sure if this is useful. 
	# date, time = zip(*[(d.date(), d.time()) for d in weather_df['DATE']])
	# weather_df = weather_df.assign(DATE=date, time=time)

	temp_df = weather_df[['DATE', 'TEMP']]
	temp_df = to_float(temp_df, 'TEMP')
	temp_df = temp_df[temp_df.TEMP != -999.99]
	temp_df.plot(x='DATE', y='TEMP')
	plt.show()

	prec_df = weather_df[['DATE', 'PREC']]
	prec_df = to_float(prec_df, 'PREC')
	prec_df = prec_df[prec_df.PREC != -999.99]
	prec_df.plot(x='DATE', y='PREC',)
	plt.show()

	# ===================================================================
	# Export cleaned datasets
	# ===================================================================
	# dir_path = "/home/dotson/Documents/cairo/"
	
	# temp_df.to_csv(dir_path + "willard_temp.csv")
	# prec_df.to_csv(dir_path + "willard_prec.csv")


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
