****************
Sunrise Examples
****************

The sunrise module has functions that generally take
inputs of time and approximates properties of the
Earth-Sun relationship.

:py:func:`sunrise.timestamp_to_hour`
------------------------------------

Example:
Take the example of six minutes into January in 2017.

>>> import pandas as pd
>>> timestamp = pd.Timestamp('2017-01-01T0006')
>>> print(timestamp)
    2017-01-01 00:06:00
>>> timestamp_to_hour(timestamp)
    24.1


:py:func:`sunrise.hour_number`
------------------------------

Example:
Take the example of six minutes into January first

>>> N = 1
>>> time = 0.1
>>> hour_number(N, time)
    24.1


:py:func:`sunrise.day_number`
-----------------------------

Example:
Midnight on day 366 of a leap year

>>> hour_number = 8784
>>> day_number(hour_number)
    366


:py:func:`sunrise.local_time`
-----------------------------

Example:
Midnight on the third day of a year.

>>> N_days = 3
>>> time = 24
>>> hour_number = hour_number(3, 24)
>>> local_time(hour_number)
    0

For a period of two days, the function local_time behaves as

>>> w = np.linspace(0,48,480).tolist()
>>> ts = [local_time(i) for i in w]
>>> plt.plot(q,ts)
>>> plt.title('Local Time Plot')
>>> plt.xlabel('Hour Number')
>>> plt.ylabel('Local Time')
>>> plt.xticks(np.arange(0, 56, step=8))
>>> plt.yticks(np.arange(0, 32, step=8))

.. image:: ../examples/plots/sunrise_plots/lt_2day.png
    :align: center


:py:func:`sunrise.frac_year`
----------------------------

Example:
Six minutes into the third day of a leap year.
It is important to remember that for a leap year you
must put the True Boolean as the second parameter.
If you want a non-leap year you can either put only
one argument or use the False Boolean.

>>> hour = hour_number(3, 0.1)
>>> frac_year(hour, True)
    -76.7172131147541
    
For a period of one year, the function frac_year behaves as

>>> w = np.linspace(0,8760,87600).tolist()
>>> ts = [frac_year(i) for i in w]
>>> plt.plot(w,ts)
>>> plt.title('Year Fraction')
>>> plt.xlabel('Hour Number')
>>> plt.xticks(np.arange(0, 9000, step=895))
>>> plt.yticks(np.arange(-77, 340, step=100))

.. image:: ../examples/plots/sunrise_plots/yf_1year.png
    :align: center


:py:func:`sunrise.declination`
------------------------------

Example:
The declination six minutes into January third of
a leap year can be calculated as follows. It is
important to remember that for a leap year you
must put the True Boolean as the second parameter.
If you want a non-leap year you can either put only
one argument or use the False Boolean.

>>> hour = hour_number(3,0.1)
>>> declination(hour, True)
    -22.81293175279647

For a period of one year, the function declination behaves as

>>> w = np.linspace(0,8760,87600).tolist()
>>> ts = [declination(i) for i in w]
>>> plt.plot(w,ts)
>>> plt.title('Declination of the Sun')
>>> plt.xlabel('Hour Number')
>>> plt.xticks(np.arange(0, 8880, step=975))
>>> plt.yticks(np.arange(-24, 28, step=8))

.. image:: ../examples/plots/sunrise_plots/dec_1year.png
    :align: center


:py:func:`sunrise.equation_of_time`
-----------------------------------

Example:
Equation of time at six minutes on January 3rd of a
leap year It is important to remember that for a
leap year you must put the True Boolean as the
second parameter. If you want a non-leap year you
can either put only one argument or use the False
Boolean.

>>> hour = hour_number(3,0.1)
>>> equation_of_time(hour, True)
    -4.684279708368122
    
For a period of two years, the function equation_of_time behaves as

>>> w = np.linspace(0,17520,87600).tolist()
>>> ts = [equation_of_time(i) for i in w]
>>> plt.plot(w,ts)
>>> plt.title('Equation of Time')
>>> plt.xlabel('Hour Number')

.. image:: ../examples/plots/sunrise_plots/et_2year.png
    :align: center


:py:func:`sunrise.local_meridian`
---------------------------------

Example:
Calculating the Local Meridian for Urbana, IL

>>> utc = -6
>>> local_meridian(utc)
    -90


:py:func:`sunrise.time_correction`
----------------------------------

Example:
To calculate the time correction at Solar Farm 1.0 in Champaign
Illinois, define the longitude, hour, and utm. Then use the
local_meridian and equation_of_time functions to calculate
et and lstm. Six minutes into a non-leap year at Solar Farm 1.0
the time_correction calculation looks like

>>> hour = 0.1
>>> lon = -88.244027
>>> utc = -6
>>> lstm = local_meridian(utc)
>>> et = equation_of_time(hour, False)
>>> time_correction(lstm, et, lon)
    3.7657778966189506
    
For a period of two years, the function time_correction behaves as

>>> w = np.linspace(0,17520,87600).tolist()
>>> lstm = -90
>>> lon = -88.244027
>>> ts = [time_correction(lstm,equation_of_time(i),
        lon) for i in w]
>>> plt.plot(z,ts)
>>> plt.title('Time Correction')
>>> plt.xlabel('Hour Number')

.. image:: ../examples/plots/sunrise_plots/tc_2year.png
    :align: center


:py:func:`sunrise.local_solar_time`
-----------------------------------

Example:
For six minutes into the third day of January of
a leap year at Solar Farm 1.0 in Urbana IL, calculate
the local solar time as follows. Remember to use
the True and False Booleans accordingly, where True
corresponds to a leap year.

>>> hour = hour_number(3,0.1)
>>> et = equation_of_time(hour, True)
>>> lon = -88.244027
>>> utc = -6
>>> lstm = local_meridian(utc)
>>> tc = time_correction(lstm, et, lon)
>>> local_time = local_time(0.1)
>>> local_solar_time(local_time, tc)
    0.16276296494364917
    
For a period of four days, the function local_solar_time behaves as

>>> w = np.linspace(0,8760,87600).tolist()
>>> lstm = -90
>>> lon = -88.244027
>>> ts = [local_solar_time(local_time(i),
        time_correction(lstm,
        equation_of_time(i),lon))
        for i in w]
>>> plt.plot(z,ts)
>>> plt.xlim(0,97)
>>> plt.title('Local Solar Time')
>>> plt.xlabel('Hour Number')

.. image:: ../examples/plots/sunrise_plots/lst_4day.png
    :align: center


:py:func:`sunrise.hour_angle`
-----------------------------

Example:
For six minutes into the third day of January of
a leap year at Solar Farm 1.0 in Urbana IL, calculate
the hour angle as follows. Remember to use
the True and False Booleans accordingly, where True
corresponds to a leap year.

>>> hour = hour_number(3,0.1)
>>> et = equation_of_time(hour, True)
>>> lon = -88.244027
>>> utc = -6
>>> lstm = local_meridian(utc)
>>> tc = time_correction(lstm, et, lon)
>>> local_time = local_time(0.1)
>>> lst = local_solar_time(local_time, tc)
>>> hour_angle(lst)
    -177.55855552584526

For a period of four days, the function hour_angle behaves as

>>> w = np.linspace(0,8760,87600).tolist()
>>> lstm = -90
>>> lon = -88.244027
>>> ts = [hour_angle(local_solar_time(local_time(i),
        time_correction(lstm,equation_of_time(i),
        lon))) for i in w]
>>> plt.plot(z,ts)
>>> plt.xlim(0,97)
>>> plt.title('Hour Angle')
>>> plt.xlabel('Hour Number')

.. image:: ../examples/plots/sunrise_plots/ha_4day.png
    :align: center


:py:func:`sunrise.solar_elevation`
----------------------------------

Example:
For six minutes into the third day of January of
a leap year at Solar Farm 1.0 in Urbana IL, calculate
the solar elevation as follows. Remember to use
the True and False Booleans accordingly, where True
corresponds to a leap year.

>>> hour = hour_number(3,0.1)
>>> et = equation_of_time(hour, True)
>>> lon = -88.244027
>>> utc = -6
>>> lstm = local_meridian(utc)
>>> tc = time_correction(lstm, et, lon)
>>> local_time = local_time(0.1)
>>> lst = local_solar_time(local_time, tc)
>>> ha = hour_angle(lst)
>>> delta = declination(hour, True)
>>> lat = 40.081798
>>> solar_elevation(ha, delta, lat)
    -72.64124833502152

For a period of one year, we can map the solar elevation and overlay the
plot of the declination to get

>>> w = np.linspace(0,8760,87600).tolist()
>>> lon = -88.244027
>>> lat = 40.081798
>>> lstm = -90
>>> ts = [solar_elevation(hour_angle(local_solar_time(local_time(i),
        time_correction(lstm,equation_of_time(i),lon))),
        declination(i),lat) for i in w]
>>>
>>> td = [declination(i) for i in w]
>>>
>>> plt.plot(z,ts)
>>> plt.plot(z,td) #adding the plot of declination to show the contrast
>>> plt.title('Solar Elevation')
>>> plt.xlabel('Hour Number')

.. image:: ../examples/plots/sunrise_plots/se_ov_dec.png
    :align: center

For a closer view of what the solar elevation plot actually looks like, we
can restrict the x-axis

>>> plt.plot(z,ts)
>>> plt.xlim(0,150) #zooming in on the solar_elevation plot

.. image:: ../examples/plots/sunrise_plots/se_zin.png
    :align: center


:py:func:`sunrise.generate_elevation_series`
--------------------------------------------

Example:
Finding the elevation angles for Solar Farm 1.0 in Urbana, IL
on the first day at time zero and six minutes later.

>>> hour_range = [0, 0.1]
>>> lat = 40.081798
>>> lon = -88.244027
>>> utc = -6
>>> generate_elevation_series(hour_range, lat, lon, utc)
    [-72.97564456089731, -72.86924538817553]
