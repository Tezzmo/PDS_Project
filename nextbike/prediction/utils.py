import os
import numpy as np


def get_ml_path():
    if os.path.isdir(os.path.join(os.getcwd(), 'ml-models')):
        return os.path.join(os.getcwd(), 'ml-models')
    elif os.path.isdir(os.path.join(os.getcwd(), "../ml-models")):
        return os.path.join(os.getcwd(), "../ml-models")
    else:
        raise FileNotFoundError


def get_prediction_path():
    if os.path.isdir(os.path.join(os.getcwd(), 'data')):
        return os.path.join(os.getcwd(), 'data')
    elif os.path.isdir(os.path.join(os.getcwd(), "../data")):
        return os.path.join(os.getcwd(), "../data")
    else:
        raise FileNotFoundError


def flat_array(inputList):
    
    temparray = np.empty([1])
    
    for row in inputList:
        row = row.reshape(-1)
        temparray = np.concatenate((temparray,row),axis=0)

    return temparray