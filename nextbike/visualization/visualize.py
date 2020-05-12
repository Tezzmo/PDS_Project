import folium
import matplotlib.pyplot as plt

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