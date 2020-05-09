from .utils import get_data_path
import pandas as pd
import os
import pickle
import numpy as np

# load raw nextbike data as csv file from designated directory
def read_file(path=os.path.join(get_data_path(), "input/inputData.csv")):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)

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
    if len(errorlines) > 0:
        print('Error at lines:')
        for line in errorlines:
            print(line)
    
    # create dataframe out of list
    dfTrip = pd.DataFrame(tripList, columns=['bNumber', 'sTime', 'eTime', 'duration', 'sLong', 'sLat', 'eLong', 'eLat',
                                             'weekend', 'bType', 'sPlaceNumber', 'ePlaceNumber'])
    
    return dfTrip

# preprocess the raw nextbike data with basic data cleaning techniques for creation of fixed stations
def preprocessStationData(df):
    df = df[['datetime', 'p_bikes', 'p_number']]
    df = df[df['p_number'] != 0]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df[df['p_number'].notna()]
    df['p_number'] = df['p_number'].astype('int64')
    
    return df

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
