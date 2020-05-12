from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from IPython.display import HTML
from .utils import get_gejson_path

import pandas as pd
import datetime
import folium
import os



#Give dfTrips including postalCode
def createTripsPerPostalCodeMap(dfTrips,month,start):

    if start == True:
        param = 'start'
    else:
        param = 'end'

    #Collecting the required Data
    dfFiltered = dfTrips[dfTrips['trip']==param][['trip','datetime','postalCode']]
    
    #Filter for month
    dfFiltered['datetime'] = pd.to_datetime(dfFiltered['datetime'])
    dfFiltered['month'] = pd.DatetimeIndex(dfFiltered["datetime"]).month
    dfFilteredRdy = dfFiltered[dfFiltered['month'] == month]

    #Count the number of starts per postalcode area
    tripsPerPostalCode = []
    groups = dfFilteredRdy.groupby('postalCode')

    for key in groups.groups.keys():
        postalCode = key
        numberOfStarts = len(groups.get_group(key))

        tripsPerPostalCode.append([postalCode,numberOfStarts])

    #Adjust to folium data-conditions
    dfTripsPerPostalCode = pd.DataFrame(tripsPerPostalCode,columns=['postalCode','numberOfStarts'])
    dfTripsPerPostalCode['postalCode'] = dfTripsPerPostalCode.postalCode.astype('str')

    #Read in gejson of postal ocde area boundaries
    geoDataPath = os.path.join(get_gejson_path(), "input/postleitzahlen-deutschland.geojson")

    map = folium.Map(location=[50.802578, 8.766052], default_zoom_start=10)
    map.choropleth(geo_data=geoDataPath,
             data=dfTripsPerPostalCode, # my dataset
             columns=['postalCode','numberOfStarts'], # zip code is here for matching the geojson zipcode, sales price is the column that changes the color of zipcode areas
             key_on='feature.properties.plz', # this path contains zipcodes in str type, this zipcodes should match with our ZIP CODE column
             fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='numberOfStarts')

    return map 