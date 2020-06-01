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

    #Collecting the required Data
    dfFiltered = dfTrips[['sTime','sPostalCode','ePostalCode']]

    if start == True:
        dfFiltered['postalCode'] = dfFiltered[['sPostalCode']]
    else:
        dfFiltered['postalCode'] = dfFiltered[['ePostalCode']]

    dfFiltered.drop(['sPostalCode','ePostalCode'],axis=1,inplace=True)
    #Filter for month
    #dfFiltered['sTime'] = pd.to_datetime(dfFiltered['sTime'])
    dfFiltered['month'] = pd.DatetimeIndex(dfFiltered["sTime"]).month
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

    fig = folium.Figure(width=500, height=500)
    map = folium.Map(location=[50.801716, 8.766453], zoom_start=11, min_zoom = 10, tiles="openstreetmap").add_to(fig)
    
    map.choropleth(geo_data=geoDataPath,
             data=dfTripsPerPostalCode, # my dataset
             columns=['postalCode','numberOfStarts'], # zip code is here for matching the geojson zipcode, sales price is the column that changes the color of zipcode areas
             key_on='feature.properties.plz', # this path contains zipcodes in str type, this zipcodes should match with our ZIP CODE column
             fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='numberOfStarts')
    
    
    display(map)