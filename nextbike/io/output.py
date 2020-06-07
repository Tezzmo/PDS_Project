from .utils import get_data_path
import os
import pickle
import pandas as pd

def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output/model.pkl"), 'wb'))


# save generated StationBikerNumberData as csv file to designated data directory
def save_StationBikeNumberData(df, path=os.path.join(get_data_path(), "output/StationBikeNumberData.csv")):
    try:
        df.to_csv(path)
    except:
        print("Data file could not be saved. Path was " + path)


# save generated tripData as csv file to designated data directory
def save_tripData(df, path=os.path.join(get_data_path(), "output/dfTrips_Saved.csv")):
    try:
        df.to_csv(path,sep=';',index=False)
    except:
        print("Data file could not be saved. Path was " + path)

# save generated StationData as csv file to designated data directory
def save_StationData(df, path=os.path.join(get_data_path(), "output/dfStations_Saved.csv")):
    try:
        df.to_csv(path,sep=';')
    except:
        print("Data file could not be saved. Path was " + path)

# save generated StationBikerNumberData as csv file to designated data directory
def save_dfBikesPerStationIndexs(df, path=os.path.join(get_data_path(), "output/dfBikesPerStationIndex_Saved.csv")):
    try:
        df.to_csv(path,sep=';')
    except:
        print("Data file could not be saved. Path was " + path)

# save generated Trips per day as csv file to designated data directory
def save_tripsPerDay(df, path=os.path.join(get_data_path(), "output/dfTripsPerDay_Saved.csv")):
    try:
        df.to_csv(path,sep=';',index=False)
    except:
        print("Data file could not be saved. Path was " + path)

# save generated Weather data as csv file to designated data directory
def save_Weather(df, path=os.path.join(get_data_path(), "output/dfWeather_Saved.csv")):
    try:
        df.to_csv(path,sep=';')
    except:
        print("Data file could not be saved. Path was " + path)



# save generated tripData as csv file to designated data directory
def save_tripDataForReues(df, path=os.path.join(get_data_path(), "input/dfTrips_Saved.csv")):
    try:
        df.to_csv(path,sep=';',index=False)
    except:
        print("Data file could not be saved. Path was " + path)

# save generated StationData as csv file to designated data directory
def save_StationDataForReues(df, path=os.path.join(get_data_path(), "input/dfStations_Saved.csv")):
    try:
        df.to_csv(path,sep=';')
    except:
        print("Data file could not be saved. Path was " + path)

# save generated StationBikerNumberData as csv file to designated data directory
def save_dfBikesPerStationIndexForReues(df, path=os.path.join(get_data_path(), "input/dfBikesPerStationIndex_Saved.csv")):
    try:
        df.to_csv(path,sep=';')
    except:
        print("Data file could not be saved. Path was " + path)

# save generated Trips per day as csv file to designated data directory
def save_tripsPerDayForReues(df, path=os.path.join(get_data_path(), "input/dfTripsPerDay_Saved.csv")):
    try:
        df.to_csv(path,sep=';',index=False)
    except:
        print("Data file could not be saved. Path was " + path)

# save generated Weather data as csv file to designated data directory
def save_WeatherForReues(df, path=os.path.join(get_data_path(), "input/dfWeather_Saved.csv")):
    try:
        df.to_csv(path,sep=';')
    except:
        print("Data file could not be saved. Path was " + path)

