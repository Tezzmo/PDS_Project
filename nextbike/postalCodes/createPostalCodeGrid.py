import math
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib


#Define the range of possible coordinates
latUpper = 50.865136
latLower = 50.733665

lngLeft = 8.605472
lngRight = 8.864199

#Define the number of 
#Refine me
limit = 12000
intervallLenght = math.floor(math.sqrt(limit))


def computeGridPoints():
    #Compute distance between points
    latStep = (latUpper - latLower)/intervallLenght
    lngStep = (lngRight - lngLeft)/intervallLenght

    #Compute the single points
    gridPointList = []
    for i in range(0,intervallLenght):
        for j in range(0,intervallLenght):
            gridPointList.append([latLower+i*latStep,lngLeft+j*lngStep])

    return gridPointList


def getPostalCodes(gridPointList):

    #Get Postalcode for every point
    postalcodeList = []
    geolocator = Nominatim(user_agent="Marburg")

    lastValid = None

    for gridPoint in gridPointList:

        try:
            postalCode = geolocator.reverse(gridPoint).raw["address"]["postcode"]

            if postalCode in [None,'None'] :
                postalcodeList.append(lastValid)

            else:
                postalcodeList.append(postalCode)
                lastValid = postalCode

        except:
            postalcodeList.append(lastValid)


    #Check if all work correctly
    if len(gridPointList)!=len(postalcodeList):
        raise Exception("Not the Same number of cluster and postalcodes")
    else:
        return postalcodeList



def saveGrid(gridPointList,postalcodeList):

    dfGripPoints = pd.DataFrame(gridPointList)
    dfPostalCode = pd.DataFrame(postalcodeList)
    
    createAndSaveNNC(dfGripPoints,dfPostalCode)

    dfMerged = pd.concat([dfPostalCode,dfGripPoints],axis=1)
    dfMerged.columns = ['PostalCode','lat','lng']
    dfMerged.to_csv('PostalCodeForCoordinates.csv',sep=';')



def createAndSaveNNC(dfGripPoints,dfPostalCode):

    nnc = KNeighborsClassifier(n_neighbors=1)
    nnc.fit(dfGripPoints,dfPostalCode)
    joblib.dump(nnc,'nearestNeighbor_PostalCode.pkl')



def createGrid():

    gridPointCoordinateList = computeGridPoints()
    gridPointPostalCodeList = getPostalCodes(gridPointCoordinateList)
    saveGrid(gridPointCoordinateList,gridPointPostalCodeList)
    createAndSaveNNC(gridPointCoordinateList,gridPointPostalCodeList)


