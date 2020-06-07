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

#Define the number of grid points
limit = 12000
#Compute number of grid points per "line"
intervallLenght = math.floor(math.sqrt(limit))


#Compute the Coordinates of the Grid points
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


#Get the postal code for the single points of the grid
def getPostalCodes(gridPointList):

    #Get Postalcode for every point
    postalcodeList = []
    geolocator = Nominatim(user_agent="Marburg")

    #If geolocater returns no result, take the last valid
    lastValid = None

    for gridPoint in gridPointList:
        try:
            #Get Postalcode for the Coordinates
            postalCode = geolocator.reverse(gridPoint).raw["address"]["postcode"]

            #If geolocater returns no result, take the last valid postal code
            if postalCode in [None,'None'] :
                postalcodeList.append(lastValid)

            #Else take the real postal code
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


#Save the grid and a nearest neigbor as csv,.pkl
def saveGrid(gridPointList,postalcodeList):

    dfGripPoints = pd.DataFrame(gridPointList)
    dfPostalCode = pd.DataFrame(postalcodeList)
    
    #Train a NN-Algo with the grid points and postalcodes
    createAndSaveNNC(dfGripPoints,dfPostalCode)

    #Merge gridpoints and postalcodes and save as csv
    dfMerged = pd.concat([dfPostalCode,dfGripPoints],axis=1)
    dfMerged.columns = ['PostalCode','lat','lng']
    dfMerged.to_csv('PostalCodeForCoordinates.csv',sep=';')


#Creates and saves a Nearest neightbor algorithm as a .pkl
def createAndSaveNNC(dfGripPoints,dfPostalCode):

    #Train the NN
    nnc = KNeighborsClassifier(n_neighbors=1)
    nnc.fit(dfGripPoints,dfPostalCode)
    #Save it
    joblib.dump(nnc,'nearestNeighbor_PostalCode.pkl')


#Serves as entry point for external use. Creates the grid and trains the NN.
def createGrid():
    #Create the grid
    gridPointCoordinateList = computeGridPoints()
    #Get postal code of single grid points
    gridPointPostalCodeList = getPostalCodes(gridPointCoordinateList)
    #Save the grid
    saveGrid(gridPointCoordinateList,gridPointPostalCodeList)
    #Save the NN
    createAndSaveNNC(gridPointCoordinateList,gridPointPostalCodeList)


