import folium
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def visualizeNumberOfBikesPerStation(pointInTime, dfStations, dfStationBikeNumber):

    m = folium.Map(location=[50.8008, 8.7667], zoom_start=13, tiles='Stamen Toner')

    bikesPerStation = dfStationBikeNumber.loc[pointInTime]
    for index, row in dfStations.iloc[1:].iterrows():     
        folium.CircleMarker(
            location=[row['pLat'], row['pLong']],
            radius=int(bikesPerStation[int(row.name)]),
            popup=row['pName'],
            color='#3186cc',
            fill=True,
            fill_color='#3186cc'
        ).add_to(m)

    return m

def visualizeMeanTripLength(df):
    # Calculate mean trip length per month, day, hour
    meanTripLengthPerMonth = df.groupby(df.sTime.dt.month).durationInSec.mean(numeric_only=False)
    meanTripLengthPerDayOfWeek = df.groupby(df.sTime.dt.dayofweek).durationInSec.mean(numeric_only=False)
    meanTripLengthPerHour = df.groupby(df.sTime.dt.hour).durationInSec.mean(numeric_only=False)
    # plot figures
    plt.rcParams["figure.figsize"][0] = 30
    plt.rcParams["figure.figsize"][1] = 5
    # Mean trip length per Month
    plt.subplot(1, 3, 1)
    plt.plot(meanTripLengthPerMonth.index, meanTripLengthPerMonth/60, 'x-')
    plt.xlabel("Per month")
    plt.ylabel('Mean trip length in minutes')
    # Mean trip length per Day
    plt.subplot(1, 3, 2)
    plt.plot(meanTripLengthPerDayOfWeek.index, meanTripLengthPerDayOfWeek/60, 'x-')
    plt.title("Mean trip length")
    plt.xlabel("Per day")
    # Mean trip length per Hour
    plt.subplot(1, 3, 3)
    plt.plot(meanTripLengthPerHour.index, meanTripLengthPerHour/60, 'x-')
    plt.xlabel("Per hour")

    return plt

def visualizeStdTripLength(df):
    stdTripLengthPerMonth = df.groupby(df.sTime.dt.month).durationInSec.std()
    stdTripLengthPerDayOfWeek = df.groupby(df.sTime.dt.dayofweek).durationInSec.std()
    stdTripLengthPerHour = df.groupby(df.sTime.dt.hour).durationInSec.std()
    # plot figures
    plt.rcParams["figure.figsize"][0] = 30
    plt.rcParams["figure.figsize"][1] = 5
    # Standard deviation of trip length per Month
    plt.subplot(1, 3, 1)
    plt.plot(stdTripLengthPerMonth.index, stdTripLengthPerMonth/60, 'x-')
    plt.xlabel("Per month")
    plt.ylabel('Standard deviation in minutes')
    # Standard deviation of trip length per Day
    plt.subplot(1, 3, 2)
    plt.plot(stdTripLengthPerDayOfWeek.index, stdTripLengthPerDayOfWeek/60, 'x-')
    plt.title("Standard deviation of trip length")
    plt.xlabel("Per day")
    # Standard deviation of trip length per Hour
    plt.subplot(1, 3, 3)
    plt.plot(stdTripLengthPerHour.index, stdTripLengthPerHour/60, 'x-')
    plt.xlabel("Per hour")

    return plt

def visualizeNumberOfTrips(df):
    numberOfTripsPerMonth = df.groupby(df.sTime.dt.month).bNumber.count()
    numberOfTripsPerDay = df.groupby(df.sTime.dt.dayofweek).bNumber.count()
    numberOfTripsPerHour = df.groupby(df.sTime.dt.hour).bNumber.count()

    # plot figures
    plt.rcParams["figure.figsize"][0] = 30
    plt.rcParams["figure.figsize"][1] = 5

    # Number of Trips per Month
    plt.subplot(1, 3, 1)
    plt.plot(numberOfTripsPerMonth.index, numberOfTripsPerMonth)
    plt.xlabel('Month')
    plt.ylabel('Number of Rentals')

    # Number of Trips per Day
    plt.subplot(1, 3, 2)
    plt.plot(numberOfTripsPerDay.index, numberOfTripsPerDay)
    plt.xlabel('Day')
    plt.title("Number of Trips in a given Time")

    # Number of Trips per Hour
    plt.subplot(1, 3, 3)
    plt.plot(numberOfTripsPerHour.index, numberOfTripsPerHour)
    plt.xlabel('Hour')

    return plt

def visualizeTripLengthBoxplots(df):
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    fig.suptitle('Trip Length in Boxplots')
    df['month'] = pd.DatetimeIndex(df["sTime"]).month
    df['day'] = pd.DatetimeIndex(df["sTime"]).day
    df['hour'] = pd.DatetimeIndex(df["sTime"]).hour

    # Boxplots per Month
    sns.boxplot(y='durationInSec', x='month', data=df, palette="colorblind", showfliers=False, ax=axes[0])
    # Boxplots per Day
    sns.boxplot(y='durationInSec', x='day', data=df, palette="colorblind", showfliers=False, ax=axes[1])
    # Boxplots per Day
    sns.boxplot(y='durationInSec', x='hour', data=df, palette="colorblind", showfliers=False, ax=axes[2])

    return sns

def visualizeWeatherData(df):
    # calculate aggregate statistics
    minTemperaturePerWeek = df.groupby(df.index.week).temperature.min()
    maxTemperaturePerWeek = df.groupby(df.index.week).temperature.max()
    meanPrecipitationPerWeek = df.groupby(df.index.week).precipitation.mean()
    # create plot
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # create plot for temperature data
    ax1.plot(minTemperaturePerWeek.index, minTemperaturePerWeek, '--k', label="Minimum temperature")
    ax1.plot(maxTemperaturePerWeek.index, maxTemperaturePerWeek, '-k', label="Maximum temperature")
    ax1.set_ylabel('Temperature (in $^\circ C$ )',size=14)
    ax1.legend(loc=2)
    # fill area between min and max line
    ax1=plt.gca()
    ax1.axis([0,52,-20,50])
    plt.gca().fill_between(minTemperaturePerWeek.index, minTemperaturePerWeek, maxTemperaturePerWeek, facecolor='red', alpha=0.1)

    # create plot for precipitation data
    ax2 = ax1.twinx()
    ax2.bar(meanPrecipitationPerWeek.index, meanPrecipitationPerWeek)
    ax2.set_ylabel('Precipitation (in mm)',size=14)
    ax2.axis([0,52,0,0.2])
    ax2.set_xlabel('Month',size=14)

    plt.title('Temperature & Precipitation in Marburg (2019)',size=14)
    plt.xticks(np.arange(0,52,4.7), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

    return plt