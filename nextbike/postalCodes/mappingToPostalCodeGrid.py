from geopy.geocoders import Nominatim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from .utils import get_ml_path
from .utils import get_gejson_path
import pandas as pd
import math
import os
import json
#from shapely.geometry import shape, Point


#Adds the postal codes to the single trips
def assignPostalCode(dfTrips):
    #Create a copy
    dfWithPostalCode = dfTrips.copy()

    #Load NN-Classifier
    path = path=os.path.join(get_ml_path(), "postalCode/nearestNeighbor_PostalCode.pkl")
    nnc = joblib.load(path , mmap_mode ='r')

    #Use NN to predict postal codes of trips
    dfWithPostalCode['sPostalCode'] = nnc.predict(dfWithPostalCode[['sLat','sLong']])
    dfWithPostalCode['ePostalCode'] = nnc.predict(dfWithPostalCode[['eLat','eLong']])

    return dfWithPostalCode


#Filter out trips, which are not in Marburg
def filterForPostalCodes(dfTrips):

    #Only use trips which start and end in marburg
    dfFilteredTrips = dfTrips[(dfTrips['sPostalCode'] == 35037)|(dfTrips['sPostalCode'] == 35039)|(dfTrips['sPostalCode'] == 35041)|(dfTrips['sPostalCode'] == 35043)]
    dfFilteredTrips = dfFilteredTrips[(dfTrips['ePostalCode'] == 35037)|(dfTrips['ePostalCode'] == 35039)|(dfTrips['ePostalCode'] == 35041)|(dfTrips['ePostalCode'] == 35043)]

    return dfFilteredTrips
    
'''
#Alternative method to assign postal codes, using geojson. Not used because of performance issues
def assignPostalCodeAlternative(dfTrips):

    dfWithPostalCode = dfTrips
    path = path=os.path.join(get_gejson_path(), "input/postleitzahlen-deutschland.geojson")
    with open(path) as f:
        js = json.load(f)

    # construct point based on lon/lat returned by geocoder

    postalCodesStart = []
    postalCodesEnd = []

    # check each polygon to see if it contains the point
    for index, row in dfWithPostalCode.iterrows():
        pointS = Point(row['sLong'],row['sLat'])
        postalCodesStart.append(check(row, pointS,js))

    # check each polygon to see if it contains the point
    for index, row in dfWithPostalCode.iterrows():
        pointE = Point(row['eLong'],row['eLat'])
        postalCodesEnd.append(check(row, pointE,js))


    dfWithPostalCode['sPostalCode'] = postalCodesStart
    dfWithPostalCode['ePostalCode'] = postalCodesEnd

    return dfWithPostalCode


def check(row,point,js):

    for feature in js['features']:
            polygon = shape(feature['geometry'])

            if polygon.contains(point):
                return feature['properties']['plz']

    return None
'''


