from .utils import get_data_path
import os
import pickle
import pandas as pd


def save_model(model):
    pickle.dump(model, open(os.path.join(get_data_path(), "output/model.pkl"), 'wb'))


def saveTrips(df):
    try:
        df.to_csv(r'./data/output/tripData.csv')
    except FileExistsError:
        print('File already exists.')
    except FileNotFoundError:
        print('Can not find the file or folder.')

