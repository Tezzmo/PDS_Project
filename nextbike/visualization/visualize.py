import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import webbrowser



# visualize number of bikes per fixed station and time
def visualizeNumberOfBikesPerStationMap(pointInTime, dfStations, dfStationBikeNumber):

    # create map
    m = folium.Map(location=[50.8008, 8.7667], zoom_start=13, tiles='Stamen Toner')

    # get number of bikes for all stations at specific time
    
    bikesPerStation = dfStationBikeNumber.loc[pointInTime]
    
    # iterrate over all stations
    for index, row in dfStations.iloc[1:].iterrows():     
        
        colorStation = ''
        
        #print(bikesPerStation.info())
        # get number of bikes for station
        # cast np.int64 to int. Else it brakes to JSON dump.
        radiusStation = int(dfStationBikeNumber.loc[(dfStationBikeNumber.index == pointInTime), [str(index)]].values[0][0])
        
        # set colorcode for marker based on number of bikes
        if  radiusStation < 11:
            colorStation = '#ff0000'
        elif 10 < radiusStation < 21:
            colorStation = '#e4e400'
        elif 20 < radiusStation < 31:
            colorStation = '#008000'
        elif 30 < radiusStation < 41:
            colorStation = '#1e90ff'
        elif radiusStation > 40:
            colorStation = '#0000ff'

        # configure and set marker to map
        folium.CircleMarker(
            location=[row['pLat'], row['pLong']],
            radius=radiusStation,
            popup=row['pName'],
            color=colorStation,
            fill=True,
            fill_color=colorStation
        ).add_to(m)
    
    # add an color legend to the map
    legend_html =   '''
                <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 130px; border:2px solid grey; z-index:9999; font-size:14px; background-color: white">
                &nbsp; <b>Color Legend</b><br>
                &nbsp; <text style='color: #ff0000;'>00 - 10 bikes/station</text><br>
                &nbsp; <text style='color: #e4e400;'>11 - 20 bikes/station</text><br>
                &nbsp; <text style='color: #008000;'>21 - 30 bikes/station</text><br>
                &nbsp; <text style='color: #1e90ff;'>31 - 40 bikes/station</text><br>
                &nbsp; <text style='color: #0000ff;'>41 - 50 bikes/station</text><br>
                </div>
                ''' 

    m.get_root().html.add_child(folium.Element(legend_html))
    print(type(m))
    #return final map
    filepath=os.path.abspath('data/output/BikesPerStationMap.html')
    m.save(filepath)
    webbrowser.open(filepath)

def visualizeNumberOfBikesPerStationBarplot(pointInTime, dfStations, dfStationBikeNumber):
    bikesPerStation = dfStationBikeNumber.loc[pointInTime].array
    stationNames = dfStations['pName'].iloc[1:].array
    plt.figure(figsize=(20,10))
    plt.bar(stationNames, bikesPerStation)
    plt.xticks(rotation=90)
    plt.show()
    
  

def visualizeMeanTripLength(df):
    # Calculate mean trip length per month, day, hour
    meanTripLengthPerMonth = df.groupby(df.sTime.dt.month).durationInSec.mean(numeric_only=False)
    meanTripLengthPerDayOfWeek = df.groupby(df.sTime.dt.dayofweek).durationInSec.mean(numeric_only=False)
    meanTripLengthPerHour = df.groupby(df.sTime.dt.hour).durationInSec.mean(numeric_only=False)
    # plot figures
    plt.rcParams["figure.figsize"][0] = 30
    plt.rcParams["figure.figsize"][1] = 15
    # Mean trip length per Month
    plt.subplot(3, 1, 1)
    plt.plot(meanTripLengthPerMonth.index, meanTripLengthPerMonth/60, 'x-')
    plt.xlabel("Per month")
    plt.title("Mean trip length")
    # Mean trip length per Day
    plt.subplot(3, 1, 2)
    plt.plot(meanTripLengthPerDayOfWeek.index, meanTripLengthPerDayOfWeek/60, 'x-')
    plt.xlabel("Per day")
    plt.ylabel('Mean trip length in minutes')
    # Mean trip length per Hour
    plt.subplot(3, 1, 3)
    plt.plot(meanTripLengthPerHour.index, meanTripLengthPerHour/60, 'x-')
    plt.xlabel("Per hour")

    return plt

def visualizeStdTripLength(df):
    stdTripLengthPerMonth = df.groupby(df.sTime.dt.month).durationInSec.std()
    stdTripLengthPerDayOfWeek = df.groupby(df.sTime.dt.dayofweek).durationInSec.std()
    stdTripLengthPerHour = df.groupby(df.sTime.dt.hour).durationInSec.std()
    # plot figures
    plt.rcParams["figure.figsize"][0] = 30
    plt.rcParams["figure.figsize"][1] = 15
    # Standard deviation of trip length per Month
    plt.subplot(3, 1, 1)
    plt.plot(stdTripLengthPerMonth.index, stdTripLengthPerMonth/60, 'x-')
    plt.xlabel("Per month")
    plt.title("Standard deviation of trip length")
    # Standard deviation of trip length per Day
    plt.subplot(3, 1, 2)
    plt.plot(stdTripLengthPerDayOfWeek.index, stdTripLengthPerDayOfWeek/60, 'x-')
    plt.xlabel("Per day")
    plt.ylabel('Standard deviation in minutes')
    # Standard deviation of trip length per Hour
    plt.subplot(3, 1, 3)
    plt.plot(stdTripLengthPerHour.index, stdTripLengthPerHour/60, 'x-')
    plt.xlabel("Per hour")

    return plt

def visualizeNumberOfTrips(df):
    numberOfTripsPerMonth = df.groupby(df.sTime.dt.month).bNumber.count()
    numberOfTripsPerDay = df.groupby(df.sTime.dt.dayofweek).bNumber.count()
    numberOfTripsPerHour = df.groupby(df.sTime.dt.hour).bNumber.count()

    # plot figures
    plt.rcParams["figure.figsize"][0] = 30
    plt.rcParams["figure.figsize"][1] = 15

    # Number of Trips per Month
    plt.subplot(3, 1, 1)
    plt.plot(numberOfTripsPerMonth.index, numberOfTripsPerMonth)
    plt.xlabel('Month')
    plt.title("Number of Trips in a given Time")

    # Number of Trips per Day
    plt.subplot(3, 1, 2)
    plt.plot(numberOfTripsPerDay.index, numberOfTripsPerDay)
    plt.xlabel('Day')
    plt.ylabel('Number of Rentals')

    # Number of Trips per Hour
    plt.subplot(3, 1, 3)
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


def visualizeDistributionOfTripsPerMonth(df):
    jan = (df.loc[(df['sTime'].dt.month==1)])["durationInSec"]
    feb = (df.loc[(df['sTime'].dt.month==2)])["durationInSec"]
    mar = (df.loc[(df['sTime'].dt.month==3)])["durationInSec"]
    apr = (df.loc[(df['sTime'].dt.month==4)])["durationInSec"]
    may = (df.loc[(df['sTime'].dt.month==5)])["durationInSec"]
    jun = (df.loc[(df['sTime'].dt.month==6)])["durationInSec"]
    jul = (df.loc[(df['sTime'].dt.month==7)])["durationInSec"]
    aug = (df.loc[(df['sTime'].dt.month==8)])["durationInSec"]
    sep = (df.loc[(df['sTime'].dt.month==9)])["durationInSec"]
    oct = (df.loc[(df['sTime'].dt.month==10)])["durationInSec"]
    nov = (df.loc[(df['sTime'].dt.month==11)])["durationInSec"]
    dec = (df.loc[(df['sTime'].dt.month==12)])["durationInSec"]

     # calculate mean and standard deviation for each month
    meanTripLengthPerMonth = (df.groupby(df.sTime.dt.month).durationInSec.mean(numeric_only=False)).to_dict()
    stdTripLengthPerMonth = (df.groupby(df.sTime.dt.month).durationInSec.std()).to_dict()

    # set size
    plt.rcParams["figure.figsize"][0] = 30
    plt.rcParams["figure.figsize"][1] = 30

    # Plot for january
    plt.subplot(4, 3, 1)
    x_jan = np.linspace(meanTripLengthPerMonth.get(1) - 3*stdTripLengthPerMonth.get(1), meanTripLengthPerMonth.get(1) + 3*stdTripLengthPerMonth.get(1), 100)
    plt.hist(jan, normed = True ,bins=100)
    plt.plot(x_jan, stats.norm.pdf(x_jan, meanTripLengthPerMonth.get(1), stdTripLengthPerMonth.get(1)))
    plt.title("January")

    # Plot for february
    plt.subplot(4, 3, 2)
    x_feb = np.linspace(meanTripLengthPerMonth.get(2) - 3*stdTripLengthPerMonth.get(2), meanTripLengthPerMonth.get(2) + 3*stdTripLengthPerMonth.get(2), 100)
    plt.hist(feb, normed = True ,bins=100)
    plt.plot(x_feb, stats.norm.pdf(x_feb, meanTripLengthPerMonth.get(2), stdTripLengthPerMonth.get(2)))
    plt.title("February")

    # Plot for march
    plt.subplot(4, 3, 3)
    x_mar = np.linspace(meanTripLengthPerMonth.get(3) - 3*stdTripLengthPerMonth.get(3), meanTripLengthPerMonth.get(3) + 3*stdTripLengthPerMonth.get(3), 100)
    plt.hist(mar, normed = True ,bins=100)
    plt.plot(x_mar, stats.norm.pdf(x_mar, meanTripLengthPerMonth.get(3), stdTripLengthPerMonth.get(3)))
    plt.title("March")

    # Plot for april
    plt.subplot(4, 3, 4)
    x_apr = np.linspace(meanTripLengthPerMonth.get(4) - 3*stdTripLengthPerMonth.get(4), meanTripLengthPerMonth.get(4) + 3*stdTripLengthPerMonth.get(4), 100)
    plt.hist(apr, normed = True ,bins=100)
    plt.plot(x_apr, stats.norm.pdf(x_apr, meanTripLengthPerMonth.get(4), stdTripLengthPerMonth.get(4)))
    plt.title("April")

    # Plot for may
    plt.subplot(4, 3, 5)
    x_may = np.linspace(meanTripLengthPerMonth.get(5) - 3*stdTripLengthPerMonth.get(5), meanTripLengthPerMonth.get(5) + 3*stdTripLengthPerMonth.get(5), 100)
    plt.hist(may, normed = True ,bins=100)
    plt.plot(x_may, stats.norm.pdf(x_may, meanTripLengthPerMonth.get(5), stdTripLengthPerMonth.get(5)))
    plt.title("May")

    # Plot for jun
    plt.subplot(4, 3, 6)
    x_jun = np.linspace(meanTripLengthPerMonth.get(6) - 3*stdTripLengthPerMonth.get(6), meanTripLengthPerMonth.get(6) + 3*stdTripLengthPerMonth.get(6), 100)
    plt.hist(jun, normed = True ,bins=100)
    plt.plot(x_jun, stats.norm.pdf(x_jun, meanTripLengthPerMonth.get(6), stdTripLengthPerMonth.get(6)))
    plt.title("June")

    # No data for july
    # Plot for july
    #plt.subplot(4, 3, 7)
    #x_jul = np.linspace(meanTripLengthPerMonth.get(7) - 3*stdTripLengthPerMonth.get(7), meanTripLengthPerMonth.get(7) + 3*stdTripLengthPerMonth.get(7), 100)
    #plt.hist(jul, normed = True ,bins=100)
    #plt.plot(x_jul, stats.norm.pdf(x_jul, meanTripLengthPerMonth.get(7), stdTripLengthPerMonth.get(7)))
    #plt.title("July")

    # Plot for august
    plt.subplot(4, 3, 8)
    x_aug = np.linspace(meanTripLengthPerMonth.get(8) - 3*stdTripLengthPerMonth.get(8), meanTripLengthPerMonth.get(8) + 3*stdTripLengthPerMonth.get(8), 100)
    plt.hist(aug, normed = True ,bins=100)
    plt.plot(x_aug, stats.norm.pdf(x_aug, meanTripLengthPerMonth.get(8), stdTripLengthPerMonth.get(8)))
    plt.title("August")

    # Plot for september
    plt.subplot(4, 3, 9)
    x_sep = np.linspace(meanTripLengthPerMonth.get(9) - 3*stdTripLengthPerMonth.get(9), meanTripLengthPerMonth.get(9) + 3*stdTripLengthPerMonth.get(9), 100)
    plt.hist(sep, normed = True ,bins=100)
    plt.plot(x_sep, stats.norm.pdf(x_sep, meanTripLengthPerMonth.get(9), stdTripLengthPerMonth.get(9)))
    plt.title("September")

    # Plot for october
    plt.subplot(4, 3, 10)
    x_oct = np.linspace(meanTripLengthPerMonth.get(10) - 3*stdTripLengthPerMonth.get(10), meanTripLengthPerMonth.get(10) + 3*stdTripLengthPerMonth.get(10), 100)
    plt.hist(oct, normed = True ,bins=100)
    plt.plot(x_oct, stats.norm.pdf(x_oct, meanTripLengthPerMonth.get(10), stdTripLengthPerMonth.get(10)))
    plt.title("October")

    # Plot for november
    plt.subplot(4, 3, 11)
    x_nov = np.linspace(meanTripLengthPerMonth.get(11) - 3*stdTripLengthPerMonth.get(11), meanTripLengthPerMonth.get(11) + 3*stdTripLengthPerMonth.get(11), 100)
    plt.hist(nov, normed = True ,bins=100)
    plt.plot(x_nov, stats.norm.pdf(x_nov, meanTripLengthPerMonth.get(11), stdTripLengthPerMonth.get(11)))
    plt.title("November")

    # Plot for december
    plt.subplot(4, 3, 12)
    x_dec = np.linspace(meanTripLengthPerMonth.get(12) - 3*stdTripLengthPerMonth.get(12), meanTripLengthPerMonth.get(12) + 3*stdTripLengthPerMonth.get(12), 100)
    plt.hist(dec, normed = True ,bins=100)
    plt.plot(x_dec, stats.norm.pdf(x_dec, meanTripLengthPerMonth.get(12), stdTripLengthPerMonth.get(12)))
    plt.title("December")

    return plt

  
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

# timeframe is the amount of hours around the cutTime and date that are rendered. Date example '2019-08-01'
def visualizeEventHeatmap(dfTrips,dfStations,date,startOrend,timeframe=12, max_val=50 ):
    if startOrend == 'start':
        Lat1 = 'sLat'
        Long1 = 'sLong'
        Lat2 = 'sLat'
        Long2 = 'sLong'
    elif startOrend == 'end':
        Lat1 = 'eLat'
        Long1 = 'eLong'
        Lat2 = 'eLat'
        Long2 = 'eLong'
    else:
        Lat1 = 'eLat'
        Long1 = 'eLong'
        Lat2 = 'sLat'
        Long2 = 'sLong'
    # Preparing the data
    movingDataS = dfTrips.loc[(dfTrips['sTime'] >= (np.datetime64(date) - np.timedelta64(timeframe,'h'))) & (dfTrips['sTime'] <= (np.datetime64(date)))]
    movingDataE = dfTrips.loc[(dfTrips['sTime'] >= (np.datetime64(date) )) & (dfTrips['sTime'] <= (np.datetime64(date) + np.timedelta64(timeframe,'h')))]
    

    heatDataS = np.c_[movingDataS[[Lat1,Long1]].values.tolist(),np.ones(len(movingDataS[Lat1]))*(1)].tolist()
    heatDataE = np.c_[movingDataE[[Lat2,Long2]].values.tolist(),np.ones(len(movingDataE[Lat2]))*(1)].tolist()
    heatDataSE = np.r_[np.c_[movingDataS[[Lat1,Long1]].values.tolist(),np.ones(len(movingDataS[Lat1]))*(1)].tolist(),np.c_[movingDataE[[Lat2,Long2]].values.tolist(),np.ones(len(movingDataE[Lat2]))*(-1)].tolist()].tolist()

    # Create the map
    m = folium.Map(location=[50.8008, 8.7667], zoom_start=13, tiles='Stamen Toner')
    # Add stations as black circle
    for index, row in dfStations.iloc[1:].iterrows():     
        folium.CircleMarker(
            location=[row['pLat'], row['pLong']],
            popup=row['pName'],
            color='black',
            fill=True,
            fill_color='#3186cc'
        ).add_to(m)

    f = folium.FeatureGroup(name='Start locations early')
    f2 = folium.FeatureGroup(name='Start locations later')
    f3 = folium.FeatureGroup(name='Difference in start locations.')
    # Add the heatmaps
    HeatMap(heatDataS, min_opacity=0.5,max_val=max_val,name='Start',radius=30).add_to(f)
    HeatMap(heatDataE, min_opacity=0.5,max_val=max_val,name='End',radius=30).add_to(f2)
    HeatMap(heatDataSE, min_opacity=0.5,max_val=max_val,name='End2',radius=30,show=False).add_to(f3)
    # Add layercontrol
    f.add_to(m)
    f2.add_to(m)
    f3.add_to(m)
    folium.LayerControl().add_to(m)
    # Add markers for the event
    folium.Marker(location=['50.80882', '8.77262'],icon=folium.Icon(color='black')).add_to(m)
    folium.Marker(location=['50.81902', '8.77439'],icon=folium.Icon(color='black')).add_to(m)
    folium.Marker(location=['50.80196', '8.75835'],icon=folium.Icon(color='black')).add_to(m)
    return m