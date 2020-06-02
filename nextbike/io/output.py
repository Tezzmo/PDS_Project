from .utils import get_data_path
import os
import pickle
import pandas as pd

def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output/model.pkl"), 'wb'))


# save generated tripData as csv file to designated data directory
def save_tripData(df, path=os.path.join(get_data_path(), "output/tripData.csv")):
    try:
        df.to_csv(path)
    except:
        print("Data file could not be saved. Path was " + path)

# save generated StationBikerNumberData as csv file to designated data directory
def save_StationBikeNumberData(df, path=os.path.join(get_data_path(), "output/StationBikeNumberData.csv")):
    try:
        df.to_csv(path)
    except:
        print("Data file could not be saved. Path was " + path)

# save generated StationData as csv file to designated data directory
def save_StationData(df, path=os.path.join(get_data_path(), "output/StationData.csv")):
    try:
        df.to_csv(path)
    except:
        print("Data file could not be saved. Path was " + path)
