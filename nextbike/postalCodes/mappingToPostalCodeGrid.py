from geopy.geocoders import Nominatim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from .utils import get_ml_path

import pandas as pd
import math
import os



def assignPostalCode(dfTrips):

    #Get only Coorinates
    dfWithPostalCode = dfTrips

    #Load NN-Classifier
    path = path=os.path.join(get_ml_path(), "postalCode/nearestNeighbor_PostalCode.pkl")
    nnc = joblib.load(path , mmap_mode ='r')
    dfWithPostalCode['sPostalCode'] = nnc.predict(dfWithPostalCode[['sLat','sLong']])
    dfWithPostalCode['ePostalCode'] = nnc.predict(dfWithPostalCode[['eLat','eLong']])

    return dfWithPostalCode
    


def filterForPostalCodes(dfTrips):

    #Only use trips which start and end in marburg
    dfFilteredTrips = dfTrips[(dfTrips['sPostalCode'] == 35037)|(dfTrips['sPostalCode'] == 35039)|(dfTrips['sPostalCode'] == 35041)|(dfTrips['sPostalCode'] == 35043)]
    dfFilteredTrips = dfFilteredTrips[(dfTrips['ePostalCode'] == 35037)|(dfTrips['ePostalCode'] == 35039)|(dfTrips['ePostalCode'] == 35041)|(dfTrips['ePostalCode'] == 35043)]

    return dfFilteredTrips

