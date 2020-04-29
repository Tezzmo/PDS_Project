from .utils import get_data_path
import pandas as pd
import os
import pickle


def read_file(path=os.path.join(get_data_path(), "input/inputData.csv")):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("Data file not found. Path was " + path)

def createTrips(df):
    # TODO automate saving? Or create extra funtion that saves the outcome.
    # TODO Biketypes
    try:
        df.dropna()
        df.drop(labels='Unnamed: 0', axis=1, inplace=True)
    except:
        print('Nothing to drop')

    df['datetime'] = pd.to_datetime(df['datetime'])
    dfTest = df
    tripList = []
    savedRow = []
    errorlines = []
    # Only creates a trip if a bike has a start and end point.
    for index, row in dfTest.iterrows():
        if row['trip'] == 'start':
            savedRow = row
        if row['trip'] == 'end':
            if savedRow['b_number'] == row['b_number']:
                trip = {'bNumber': row['b_number'], 'sTime': savedRow['datetime'],
                        'eTime': row['datetime'], 'duration': (row['datetime'] - savedRow['datetime']),
                        'sLong': savedRow['p_lng'], 'sLat': savedRow['p_lat'],
                        'eLong': row['p_lng'], 'eLat': row['p_lat'],
                        'weekend': (lambda x: 1 if x > 4 else 0)(row['datetime'].dayofweek)}
                tripList.append(trip)
            else:
                errorlines.append(index)
    if len(errorlines) > 0:
        print('Error at lines:')
        for line in errorlines:
            print(line)
    dfTrip = pd.DataFrame(tripList, columns=['bNumber', 'sTime', 'eTime', 'duration', 'sLong', 'sLat', 'eLong', 'eLat',
                                             'weekend'])
    return dfTrip

def read_model():
    path = os.path.join(get_data_path(), "output/model.pkl")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
