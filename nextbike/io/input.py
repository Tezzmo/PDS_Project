from .utils import get_data_path
import pandas as pd
import os
import pickle
import numpy as np
import datetime


#region read-in

# load raw nextbike data as csv file from designated directory
def read_file(path=os.path.join(get_data_path(), "input/inputData.csv")):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)

# load temperature data as csv file from designated directory
def read_tempData(path=os.path.join(get_data_path(), "input/air_temperature.txt")):
    try:
        df = pd.read_csv(path, sep=';')
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)

# load precipitation data as csv file from designated directory
def read_precData(path=os.path.join(get_data_path(), "input/precipitation.txt")):
    try:
        df = pd.read_csv(path, sep=';')
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)



def readFinalTrips(path=os.path.join(get_data_path(), "input/trips.csv")):
    try:
        df = pd.read_csv(path, sep=';')

        df['sTime'] = pd.to_datetime(df['sTime'])
        df['eTime'] = pd.to_datetime(df['eTime'])
        df['duration'] = pd.to_timedelta(df['duration'])
   
        df.drop('Unnamed: 0',axis=1,inplace=True)


        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path) 

#endregion

#region datapreparation
def getWeatherData():
    # load temperature and precipitation data
    dfTemperature = read_tempData()
    dfPrecipitation = read_precData()

    # drop unneccessary columns
    dfTemperature.drop(columns=['STATIONS_ID', 'QN', 'PP_10', 'TT_10', 'RF_10', 'TD_10', 'eor'], inplace = True, axis=1)
    dfPrecipitation.drop(columns=['STATIONS_ID', 'QN', 'RWS_DAU_10', 'RWS_IND_10', 'eor'], inplace = True, axis=1)

    # convert string to datetime
    dfTemperature['MESS_DATUM'] = pd.to_datetime(dfTemperature['MESS_DATUM'], format = '%Y%m%d%H%M')
    dfPrecipitation['MESS_DATUM'] = pd.to_datetime(dfPrecipitation['MESS_DATUM'], format = '%Y%m%d%H%M')

    # rename columns
    dfTemperature.rename(columns={'MESS_DATUM': 'date', 'TM5_10': 'temperature'}, inplace=True)
    dfPrecipitation.rename(columns={'MESS_DATUM': 'date', 'RWS_10': 'precipitation'}, inplace=True)

    # filter for year 2019
    dfTemperature = dfTemperature[(dfTemperature['date'].dt.year == 2019)]
    dfPrecipitation = dfPrecipitation[(dfPrecipitation['date'].dt.year == 2019)]

    # set index
    dfTemperature.set_index('date', inplace = True)
    dfPrecipitation.set_index('date', inplace = True)

    # combine different weather dataframes based on time index
    dfWeather = pd.concat([dfTemperature, dfPrecipitation], axis = 1)

    return dfWeather

# preprocess the raw nextbike data with basic data cleaning techniques for creation of trips
def preprocessData(df):
    # drop empty rows and unneccessary columns
    df.dropna(inplace=True)
    df.drop(columns=['Unnamed: 0', 'p_spot', 'p_place_type', 'p_uid', 'p_bikes', 'p_bike'], inplace=True)

    # drop rows which are not a start or end of a trip
    df = df[(df['trip'] == 'start') |  (df['trip'] == 'end')]

    # convert column datatype
    df['p_number'] = df.p_number.astype('int64')

    return df


# preprocess the raw nextbike data with basic data cleaning techniques for creation of fixed stations
def preprocessStationData(df):
    df = df[['datetime', 'p_bikes', 'p_number']]
    df = df[df['p_number'] != 0]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['p_number'].notna()]
    df['p_number'] = df['p_number'].astype('int64')
    
    return df

#endregion

#region create dataframes

# create dataframe with number, name and coordinates of fixed stations
def createStations(df):
    # generate array of unique place numbers
    pNumbersUnique = df.p_number.unique()

    # generate array of unique place names in the same order
    pNamesUnique = []    
    for i in pNumbersUnique:
        pName = df[df['p_number'] == i].p_name.unique()[0]
        pNamesUnique.append(pName)

    # generate array of unique place latitude coordinates in the same order
    pLatUnique = []    
    for i in pNumbersUnique:
        pLat = df[df['p_number'] == i].p_lat.unique()[0]
        pLatUnique.append(pLat)

    # generate array of unique place longitude coordinates in the same order
    pLongUnique = []    
    for i in pNumbersUnique:
        pLong = df[df['p_number'] == i].p_lng.unique()[0]
        pLongUnique.append(pLong)

    # create dataframe
    df = pd.DataFrame()
    df['pNumber'] = pNumbersUnique
    df['pName'] = pNamesUnique
    df['pLat'] = pLatUnique
    df['pLong'] = pLongUnique
    
    # set dataframe index and manipulate one column
    df = df.set_index('pNumber')
    df.at[0, 'pName'] = 'no fixed station'
    df.at[0, 'pLat'] = np.nan
    df.at[0, 'pLong'] = np.nan
    df = df.sort_index()

    # replace special characters in station names
    df['pName'] = df['pName'].str.replace('ä', 'ae')
    df['pName'] = df['pName'].str.replace('ü', 'ue')
    df['pName'] = df['pName'].str.replace('ö', 'oe')
    df['pName'] = df['pName'].str.replace('Ä', 'Ae')
    df['pName'] = df['pName'].str.replace('Ü', 'Ue')
    df['pName'] = df['pName'].str.replace('Ö', 'Oe')
    df['pName'] = df['pName'].str.replace('ß', 'ss')
    df['pName'] = df['pName'].str.replace('é', 'e')

    return df

# create dataframe which contains the nextbike data in trip format
def createTrips(df):
    # cast datatype to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    dfTest = df
    tripList = []
    savedRow = []
    errorlines = []
    
    # iterate over all rows of the dataframe
    for index, row in dfTest.iterrows():
        
        # check if trip is start ort end
        if row['trip'] == 'start':
            savedRow = row
        if row['trip'] == 'end':
            
            # check if trip start and end are in following rows have the same b_number
            if savedRow['b_number'] == row['b_number']:
                
                # create trip and append it
                trip = {'bNumber': row['b_number'], 'sTime': savedRow['datetime'],
                        'eTime': row['datetime'], 'duration': (row['datetime'] - savedRow['datetime']),
                        'sLong': savedRow['p_lng'], 'sLat': savedRow['p_lat'],
                        'eLong': row['p_lng'], 'eLat': row['p_lat'],
                        'weekend': (lambda x: True if x > 4 else False)(row['datetime'].dayofweek),
                        'bType': row['b_bike_type'], 'sPlaceNumber': savedRow['p_number'], 'ePlaceNumber': row['p_number']}
                tripList.append(trip)
            else:
                
                # save rows with errors due to not fitting b_number
                errorlines.append(index)
            
            # clean savedRow
            savedRow['b_number'] = -1
    
    # print rows with errors
    # if len(errorlines) > 0:
    #     print('Error at lines:')
    #     for line in errorlines:
    #         print(line)
    
    # create dataframe out of list
    dfTrip = pd.DataFrame(tripList, columns=['bNumber', 'sTime', 'eTime', 'duration', 'sLong', 'sLat', 'eLong', 'eLat',
                                             'weekend', 'bType', 'sPlaceNumber', 'ePlaceNumber'])
    
    return dfTrip


#Create a DF contains date and trips per date plus weather data
def createTripsPerDay(dfTrips,dfWeather):

    #Prepare delivered dataframes
    dfWeather = dfWeather.reset_index()
    dfWeather['date'] = pd.to_datetime(dfWeather['date'])
    dfWeather['sDate'] = dfWeather['date'].dt.date
    dfWeather.drop('date',axis=1,inplace=True)

    dfTrips['sDate'] = dfTrips['sTime'].dt.date

    #Create a df with date as key and # of trips per day
    tripsPerDay = []

    for date in dfTrips['sDate'].unique():
        tripsPerDay.append([date,len(dfTrips[dfTrips['sDate']==date])])

    tripsPerDay = pd.DataFrame(tripsPerDay,columns=['date','tripsPerDay'])

    #Create a df with date as key and weather data
    weatherPerDay = []

    for date in dfWeather['sDate'].unique():
        df=dfWeather[dfWeather['sDate']==date]
        weatherPerDay.append([date,df['temperature'].max(),df['temperature'].mean(),df['temperature'].min(),df['precipitation'].mean()])

    weatherPerDay = pd.DataFrame(weatherPerDay,columns=['date','temperatureMAX','temperatureAVG','temperatureMIN','precipitationAVG'])

    #Join both on key
    mergedDf = tripsPerDay.join(weatherPerDay.set_index('date'), on='date')
    mergedDf['date'] = pd.to_datetime(mergedDf['date'])

    #create new features
    mergedDf['day'] = mergedDf['date'].dt.day
    mergedDf['month'] = mergedDf['date'].dt.month
    mergedDf['dayOfWeek'] = mergedDf['date'].dt.dayofweek

    mergedDf.drop(['date'],axis=1,inplace=True)
    mergedDf.dropna(inplace=True)
    dfTripsPerDay = mergedDf

    return dfTripsPerDay


# create a datetime index based on minutes as intervall which contains the available number of bikes per fixed station
def createBikeNumberPerStationIndex(df):
    dfStations = pd.DataFrame({'datetime': pd.date_range('2019-01-01', '2020-01-01', freq='min', closed='left')})
    dfStations = dfStations.set_index('datetime')

    stations = sorted(df['p_number'].unique())

    for i in stations:
        dfOneStation = df[df['p_number'] == i]
        dfOneStation = dfOneStation.sort_values(by='datetime')
        dfOneStation = dfOneStation.drop_duplicates(subset='datetime', keep='first')
        dfOneStation = dfOneStation.drop(columns=['p_number'])
        dfOneStation = dfOneStation.rename(columns={'p_bikes': i})
        dfOneStation = dfOneStation.set_index('datetime')
        dfStations = dfStations.join(dfOneStation, how='left')

    dfStations.iloc[0] = 0
    dfStations = dfStations.fillna(method='ffill', axis='index')
    dfStations = dfStations.astype('int64')

    return dfStations


def read_model():
    path = os.path.join(get_data_path(), "output/model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# drop outliers
def drop_outliers(df):
    
    # add column with durationInSec
    df["durationInSec"] = df["duration"].dt.total_seconds().astype(int)
    
    # calculate mean and standard deviation for each day and month
    meanTripLengthPerDay = (df.groupby(df.sTime.dt.date).durationInSec.mean(numeric_only=False)).to_dict()
    stdTripLengthPerDay = (df.groupby(df.sTime.dt.date).durationInSec.std()).to_dict()
    meanTripLengthPerMonth = (df.groupby(df.sTime.dt.month).durationInSec.mean(numeric_only=False)).to_dict()
    stdTripLengthPerMonth = (df.groupby(df.sTime.dt.month).durationInSec.std()).to_dict()
    
    # initialize data
    date = datetime.date(2019, 1, 20)
    month = date.month
    meanDay = meanTripLengthPerDay.get(date)
    stdDay = stdTripLengthPerDay.get(date)
    meanMonth = meanTripLengthPerMonth.get(month)
    stdMonth = stdTripLengthPerMonth.get(month)
    indexList = []
    
    for index, row in df.iterrows():
        
        newDate = row['sTime'].date()
        
        # check if new mean and std need to be loaded
        if (newDate != date):
            date = newDate
            month = date.month
            meanDay = meanTripLengthPerDay.get(date)
            stdDay = stdTripLengthPerDay.get(date)
            meanMonth = meanTripLengthPerMonth.get(month)
            stdMonth = stdTripLengthPerMonth.get(month)
        
        # drop outliers that are not within the range of mean +- standard deviation
        if (row['durationInSec'] < (meanMonth - 0.5 * stdMonth) or row['durationInSec'] > (meanMonth + 0.5 * stdMonth)):
            indexList.append(index)
        elif (row['durationInSec'] < (meanDay - 1 * stdDay) or row['durationInSec'] > (meanDay + 1 * stdDay)):
            indexList.append(index)
    
    df.drop(indexList, inplace=True)
    
    return df