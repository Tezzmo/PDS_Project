from .utils import get_data_path
import pandas as pd
import os
import pickle

# load raw nextbike data as csv file from designated directory
def read_file(path=os.path.join(get_data_path(), "input/inputData.csv")):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)

# preprocess the raw nextbike data with basic data cleaning techniques
def preprocessData(df):
    # drop empty rows and unneccessary columns
    df.dropna(inplace=True)
    df.drop(columns=['Unnamed: 0', 'p_spot', 'p_place_type', 'p_uid', 'p_bikes', 'p_bike'], inplace=True)

    # drop rows which are not a start or end of a trip
    df = df[(df['trip'] == 'start') |  (df['trip'] == 'end')]

    # convert column datatype
    df['p_number'] = df.p_number.astype('int64')

    return df

# create a dictionary which stores the names of the fixed stations for the p_number as key
def createStationNameDictionary(df):
    # generate array of unique place numbers
    pNumbersUnique = df.p_number.unique()

    # generate array of unique place names in the same order
    pNamesUnique = []    
    for i in pNumbersUnique:
        pName = df[df['p_number'] == i].p_name.unique()[0]
        pNamesUnique.append(pName)

    # generate dictionary
    pNameNumberDict = dict(zip(pNumbersUnique, pNamesUnique))

    pNameNumberDict[0] = 'Kein offizieller Standort'

    return pNameNumberDict

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

def read_model():
    path = os.path.join(get_data_path(), "output/model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
