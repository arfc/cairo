import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# coordinates for solar farm 1.0
lat = 40.081798
lon = -88.244027
utc = -6


def timestamp_to_hour(timestamp):
    """
    This function returns the hour number
    of a particular timestamp.

    Parameters
    ----------
    timestamp : pandas Timestamp
        The time stamp of interest, must be in 24-hour.

    Returns
    -------
    hour : float
        The number of hours since 12 AM on January first.
    """
    minutes_of_hour = timestamp.minute / 60
    seconds_of_hour = timestamp.second / 3600
    hour_frac = timestamp.hour + minutes_of_hour + seconds_of_hour
    day_number = timestamp.dayofyear

    hour = hour_number(day_number, hour_frac)

    return hour


def hour_number(N, time):
    """
    Takes the day number and time (in hours) and
    converts to hour number.

    Parameters
    ----------
    N : integer
        The day number
    time : float
        The time in hours (24-hour clock)

    Returns
    -------
    hour : float
        The hour number
    """
    hour = N * 24 + time

    return hour


def day_number(hour_number):
    """
    Returns the day_number given a particular hour_number

    Parameters
    ----------
    hour_number : float
        The number of hours since Jan first of a given year

    Returns
    -------
    day_number : int
        The hour number
    """
    N = (hour_number / 8760) * 365

    return int(N)


def local_time(hour_number):
    """
    Takes the hour number, and converts to local time.

    Parameters
    ----------
    hour_number : float
        The number of hours since Jan first of a given year

    Returns
    -------
    time : float
        The number of hours since 12 AM on any given day
    """
    N = day_number(hour_number)

    time = hour_number - N * 24

    return time


def frac_year(hour, leap_year=False):
    """
    This function calculates the fraction of the year.

    Parameters
    ----------
    hour : integer, float
        The hour of the day. Could be a decimal value.
    leap_year : boolean
        Indicates if the year is a leap year. Default
        is False.

    Returns
    -------
    B : float
        The fraction of the year
    """
    if leap_year:
        n_days = 366
    else:
        n_days = 365

    B = (hour - 1944) / 24 * 360 / n_days
    return B


def declination(hour, leap_year=False):
    """
    This function calculates the declination angle of the sun.

    Parameters
    ----------
    hour : integer, float
        The hour of the day. Could be a decimal value.
    leap_year : boolean
        Indicates if the year is a leap year. Default
        is False.

    Returns
    -------
    delta : float
        The declination at the given hour
    """
    B = frac_year(hour, leap_year)
    delta = 23.44 * np.sin((np.pi / 180) * B)

    return delta


def equation_of_time(hour, leap_year=False):
    """
    This function calculates the equation of time. The equation
    of time gives the difference between the solar time and wall
    clock time, effectively correcting for the eccentricity of
    Earth's orbit.

    Parameters
    ----------
    hour : float
        The number of hours since midnight, January first.

    Returns
    -------
    et : float
        The time in minutes
    """
    B = frac_year(hour, leap_year)
    et = 9.87 * np.sin(2 * B * (np.pi / 180)) - 7.53 * \
        np.cos(B * (np.pi / 180)) - 1.5 * np.sin(B * (np.pi / 180))

    return et


def local_meridian(utc=utc):
    """
    This function calculates the local standard time meridian,
    LSTM. The LSTM is a reference meridian for given time zones.
    LSTM is calculated according to the equation:
        LSTM = 360/24*UTC

    UTC < 0 gives the answer in degrees West.
    UTC > 0 gives the answer in degrees East.

    Parameters
    ----------
    utc : integer
        UTC is the "coordinated universal time." This parameter
        gives the UTC offset for a particular timezone.
        E.g. Chicago time is UTC-5, so we take utc = 5.

    Returns
    -------
    lstm : float
        The LSTM in degrees
    """
    lstm = 15 * utc

    return lstm


def time_correction(lstm, et, lon=lon):
    """
    The time correction factor adjusts for differences in
    Local Solar Time (LST) and the location of interest, at
    a given longitude. There is a factor of 4 due to the speed
    of Earth's rotation.

    Parameters
    ----------
    lstm : float
        The local standard time meridian.
    et : float
        The equation of time.
    lon : float
        The longitude of the location of interest.

    Returns
    -------
    tc : float
        Time correction factor in minutes.
    """
    tc = 4 * (lon - lstm) + et

    return tc


def local_solar_time(local_time, tc):
    """
    Corrects the local time to give the local solar time.

    Parameters
    ----------
    local_time : float
        The local time in hours
    tc : the time correction factor

    Returns
    -------
    lst : float
        The local solar time in hours.
    """

    lst = local_time + tc / 60

    return lst


def hour_angle(lst):
    """
    This function calculates the hour angle of the sun.
    The hour angle is the number of degrees the sun has
    moved across the sky.

    Parameters
    ----------
    lst : float
        The local solar time

    Returns
    -------
    ha : float
        The hour angle in minutes
    """
    ha = (15 * (lst - 12))

    return ha


def solar_elevation(ha, delta, lat=lat):
    """
    Calculate the solar elevation for a given hour angle,
    latitude, and declination.

    Parameters
    ----------
    ha : float
        The hour angle
    delta : float
        The declination angle
    lat : float
        The latitude at the location of interest.

    Returns
    -------
    alpha : float
        The elevation angle
    """

    sin_term = np.sin(delta * np.pi / 180) * np.sin(lat * np.pi / 180)
    cos_term = np.cos(delta * np.pi / 180) * np.cos(lat * \
                      np.pi / 180) * np.cos(ha * np.pi / 180)
    alpha = np.arcsin((sin_term + cos_term)) * 180 / np.pi

    return alpha


def generate_elevation_series(
        hour_range,
        lat=lat,
        lon=lon,
        utc=utc,
        timestamps=False):
    """
    Creates a time series of elevation angles for a given
    set of hours or Pandas timestamps.

    Parameters
    ----------
    hour_range : numpy array or pandas series of Timestamps
        The time period of desired solar elevation angles.
        Default is a list of hours, a pandas series of
        timestamps can also be accepted if the timesteps
        argument is set to True.
    lat : float
        The latitude of the location of interest
    lon : float
        The longitude of the location of interest
    utc : integer
        The time shift from the Coordinated Universal Time.
        Western longitudes have UTC < 0. Eastern longitudes
        have UTC > 0.
    timestamps : boolean
        Specifies the type of the hour_range.
        Default is False (hour_range is array of float or int).
        True if hour_range is a pandas series of Timestamps.

    Returns
    -------
    elevation_angles : list
        The list of elevation angles corresponding to each
        hour or timestamp. Units in degrees.
    """
    elevation_angles = []

    for hour in hour_range:
        if timestamps:
            hour = timestamp_to_hour(hour)
        time = local_time(hour)
        et = equation_of_time(hour)
        dec = declination(hour)
        LSTM = local_meridian(utc)
        tc = time_correction(LSTM, et)
        lst = local_solar_time(time, tc)
        ha = hour_angle(lst)
        elangle = solar_elevation(ha, dec)
        elevation_angles.append(elangle)

    return elevation_angles


if __name__ == "__main__":
    time = 0  # hours since midnight
    N = 1

    hour = hour_number(N, time)

    et = equation_of_time(hour)

    lstm = local_meridian()

    tc = time_correction(lstm, et)

    lst = local_solar_time(time, tc)

    ha = hour_angle(lst)

    dec = declination(hour)

    elangle = solar_elevation(ha, dec)

    print(f"The time is: {time}")
    print(f"The hour number is: {hour}")
    print(f"The equation of time is: {et}")
    print(f"The local standard time meridian is: {lstm}")
    print(f"The time correction factor is: {tc}")
    print(f"The local solar time is: {lst}")
    print(f"The hour angle is: {ha}")
    print(f"The elevation angle is: {elangle}")

    rise = 12 - (180 / np.pi) * (1 / 15) * np.arccos(-np.sin(
        lat * np.pi / 180) * np.sin(
        dec * np.pi / 180) / (np.cos(lat * np.pi / 180) * np.cos(
            dec * np.pi / 180))) - tc / 60
    print(rise)

    dates = pd.date_range(start='1/1/2015', end='7/1/2019', freq='h')[:-2]

    # t = np.arange(0, 8760, 1)
    # elevation = generate_elevation_series(t)
    elevation = generate_elevation_series(dates, timestamps=True)

    # plt.figure(figsize=(12, 9), facecolor='w')
    # plt.ylabel("Solar Elevation Angle in Degrees")
    # plt.xlabel("Hours Since Start Date")
    # # plt.plot(t, elevation)
    # plt.plot(dates, elevation)
    # plt.show()

    # data = {'Time':dates, 'angle':elevation}
    # df = pd.DataFrame(data)
    # print(df)
    # df.to_csv('solar_elevation_demand.csv')
