# README for Data


### UIUC Data

The data included in this folder related to UIUC is

|Name|Filename|Description|
|:-----|:------|:-----------|
|Railsplitter Wind Farm|railspliter_data.csv|Power supplied to UIUC from Railsplitter Wind Farm in Lincoln, IL|
|Lincoln Weather| lincoln_weatherdata.csv| Weather data from Lincoln County Airport near Railsplitter Wind Farm|
UIUC Electricity Demand | uiuc_demand_data.csv | The total electricity demand on the UIUC campus.
|UIUC Solar Farm 1.0 Data|solarfarm_data.csv| Data from UIUC Solar Farm 1.0 with hourly resolution


Weather data for Champaign and Lincoln county can be accessed through the
NOAA [Local Climate Data Tool](https://www.ncdc.noaa.gov/cdo-web/datatools/lcd).
Wind farm and demand data are not publicly available. The UIUC Solar Farm 1.0
data can be accessed through the [solar farm dashboard](go.illinois.edu/solar).


### Solar Irradiance Data

The solar energy data from NREL is from the "Physical Solar Model version 3"
(PSM3). It has 30-minute resolution and the following columns.

#### Columns:
  - Pressure [mbar]
  - Temperature [deg C]
  - DHI [w/m^2]
  - DNI [w/m^2]
  - GHI [w/m^2]
  - Albedo [-]
  - Relative Humidity [%]
  - Wind Speed [m/s]

|Location|Coordinates|Year Span|Time Resolution|
|:------ |:----------:|-----:|-------:|
|Champaign, IL|40.09, -88.26| 2010-2019| 30 minute|
|Dallas, TX|32.77, -96.82|2010-2019|30 minute|
|Mansfield, OH| 40.77, -82.50|2010-2019|30 minute|
|Barstow, CA| 35.69, -105.94|2010-2019|30 minute|
|Santa Fe, NM| 34.89, -117.06|2010-2019|30 minute|
|Syracuse, NY| 43.05, -76.18|2010-2019|30 minute|

The data can be accessed from the [National Solar Radiation Database](https://maps.nrel.gov/nsrdb-viewer/) (NSRDB).


### Wind Speed Data (100m height)

The wind speed data from NREL is from the Wind Integration National Database
(WIND) Toolkit. It has 5-minute time resolution and the following columns:

#### Columns:
  - Wind speed at 100m [m/s]
  - Air temperature at 100m [C]


|Location|Coordinates|Year Span|Time Resolution|
|:------ |:----------:|-----:|-------:|
|Lincoln, IL|40.22, -89.35| 2007-2012|5 minute|
|Dallas, TX|32.79, -96.79|2007-2012|5 minute|
|Mansfield, OH| 40.76, -82.52|2007-2012|5 minute|
|Barstow, CA| 35.68, -105.91|2007-2012|5 minute|
|Santa Fe, NM| 34.89, -117.04|2007-2012|5 minute|
|Syracuse, NY| 43.07, -76.17|2007-2012|5 minute|

The data can be accessed from the [Wind Integration National Database](http://maps.nrel.gov/wind-prospector) (WIND) Toolkit.
