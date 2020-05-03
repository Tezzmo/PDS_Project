# %%
import math
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.neighbors import KNeighborsClassifier


# %%
#Define the range of possible coordinates
latUpper = 50.9213
latLower = 50.6821

lngLeft = 8.50736
lngRight = 9.02556 

#Define the number of 
#Refine me
limit = 100
intervallLenght = math.floor(math.sqrt(limit))

#Compute distance between points
latStep = (latUpper - latLower)/intervallLenght
lngStep = (lngRight - lngLeft)/intervallLenght


# %%
#Compute the single points
clusterList = []
for i in range(0,intervallLenght):
    for j in range(0,intervallLenght):
        clusterList.append([latLower+i*latStep,lngLeft+j*lngStep])



# %%
#Get Postalcode for every point
postalcodeList = []
geolocator = Nominatim(user_agent="Marburg")

counter = 0

for cluster in clusterList:
    counter += 1

    lastValid = None
    
    try:
        postalCode = geolocator.reverse(cluster).raw["address"]["postcode"]

        if postalCode is None:
            postalcodeList.append(lastValid)

        else:
            postalcodeList.append(postalCode)
            lastValid = postalCode

    except:
        postalcodeList.append(lastValid)

    print(counter)


#%%
#Check if all work correctly
if len(clusterList)!=len(postalcodeList):
    raise Exception("Not the Same number of cluster and postalcodes")


# %%
for i,code in enumerate(postalcodeList):
    if code is None:
        postalcodeList[i] = postalcodeList[i-1]

# %%
nnClassifier = KNeighborsClassifier(n_neighbors=1)
nnClassifier.fit(clusterList,postalcodeList)

# %%
