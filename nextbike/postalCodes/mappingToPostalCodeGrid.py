from geopy.geocoders import Nominatim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from .utils import get_ml_path

import pandas as pd
import math
import os



def assignPostalCode(dfInput):

    #Get only Coorinates
    dfWithPostalCode = dfInput

    #Load NN-Classifier
    path = path=os.path.join(get_ml_path(), "postalCode/nearestNeighbor_PostalCode.pkl")
    nnc = joblib.load(path , mmap_mode ='r')
    dfWithPostalCode['sPostalCode'] = nnc.predict(dfWithPostalCode[['sLat','sLong']])
    dfWithPostalCode['ePostalCode'] = nnc.predict(dfWithPostalCode[['eLat','eLong']])

    return dfWithPostalCode
    

