import numpy as np
from pandas import date_range
import jdcal
import datetime as dt
import matplotlib.pyplot as plt


# coordinates for solar farm 1.0
lat = 40.081798
lon = -88.244027

def frac_year(hour, leap_year=False):
    """
    This function calculates the fraction of the year.

    Parameters:
    -----------
    hour : integer, float
        The hour of the day. Could be a decimal value.
    leap_year : boolean
        Indicates if the year is a leap year. Default
        is False.

    Returns:
    --------
    B : float
        The fraction of the year
    """
    if leap_year:
        n_days = 365
    else:
        n_days = 364

    B = (h-1944)/24*360/n_days

    return B

def declination(hour, leap_year=False):
    """
    This function calculates the declination angle of the sun.
    Parameters:
    -----------
    hour : integer, float
        The hour of the day. Could be a decimal value.
    leap_year : boolean
        Indicates if the year is a leap year. Default
        is False.

    Returns:
    --------
    delta : float
        The declination at the given hour
    """
    B = frac_year(hour)
    delta = 23.44*np.sin((np.pi/180)*B)

    return delta

def equation_of_time(hour):
    """
    This function calculates the equation of time. The equation
    of time gives the difference between the solar time and wall
    clock time.

    Parameters:
    -----------
    hour : float
        The number of hours since midnight, January 1st.

    Returns:
    --------
    et : float
        The time in minutes
    """
    B = frac_year(hour)
    et = 9.87*np.sin(2*B*(np.pi/180)) - 7.53*np.cos(B*(np.pi/180)) - 1.5*np.cos(B*(np.pi/180))

    return et

def hour_angle(hour, lat=lat, lon=lon):
    """
    This function calculates the hour angle of the sun.

    Parameters:
    -----------
    hour : float
        The number of hours since midnight, January 1st.

    lat : float
        The northern latitude of interest.
    lon : float
        The longitude of interest.

    Returns:
    --------
    ha : float
        The hour angle in minutes
    """
    to_rads = np.pi/180

    delta = declination(hour)

    num = (np.sin(-0.83*to_rads) - np.sin(lat*to_rads)*np.sin(delta*to_rads))
    den = (np.cos(lat*to_rads)*np.cos(delta*to_rads))

    ha = np.arcsin(num/den)

    # altitude = np.arcsin(np.sin(delta)*np.sin(lat)+np.cos(delta*to_rads)*np.cos(ha)*np.cos(lat*to_rads))
    # ha = (2/15)*np.arccos(-np.tan(lat*to_rads)*np.tan(delta*to_rads))

    return ha

dec = []
et = []
ha = []
for h in range(1,365*24,1):
        # print(n,h, frac_year(n,h))
        dec.append(declination(h))
        et.append(equation_of_time(h))
        ha.append(hour_angle(h))
# plt.plot(range(1,365*24,1)[:len(dec)], dec)
# plt.plot(range(1,365*24,1)[:len(dec)], et)
plt.plot(range(1,365*24,1)[:len(ha)], ha)
plt.show()
date = dt.date(2020, 9, 26)
print(date)
print(date.toordinal())


dates = date_range(start='01/01/2015', end='01/01/2020', freq='H')

julian_dates = [stamp.toordinal() for stamp in dates]

# print(julian_dates)
