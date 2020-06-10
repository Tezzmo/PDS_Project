from sklearn.neighbors import NearestNeighbors
import joblib
from .utils import get_gejson_path
import pandas as pd
import datetime
import folium
import os
import webbrowser


#Create a map, showing the starts or ends per postal code area in a month
def createTripsPerPostalCodeMap(dfTrips,month,start):

    #Collecting the required Data
    dfFiltered = dfTrips[['sTime','sPostalCode','ePostalCode']]

    #Chose if the end or start point of trips are wanted
    if start == True:
        dfFiltered['postalCode'] = dfFiltered[['sPostalCode']]
    else:
        dfFiltered['postalCode'] = dfFiltered[['ePostalCode']]

    #Drop this, because universal attr. was created before
    dfFiltered.drop(['sPostalCode','ePostalCode'],axis=1,inplace=True)

    #Filter for the wanted month
    dfFiltered['month'] = pd.DatetimeIndex(dfFiltered["sTime"]).month
    dfFilteredRdy = dfFiltered[dfFiltered['month'] == month]

    #Get unique postal codes
    tripsPerPostalCode = []
    groups = dfFilteredRdy.groupby('postalCode')

    #Count the number of starts per postalcode area using group by
    for key in groups.groups.keys():
        postalCode = key
        numberOfStarts = len(groups.get_group(key))
        tripsPerPostalCode.append([postalCode,numberOfStarts])

    #Adjust to folium data-conditions
    dfTripsPerPostalCode = pd.DataFrame(tripsPerPostalCode,columns=['postalCode','numberOfStarts'])
    dfTripsPerPostalCode['postalCode'] = dfTripsPerPostalCode.postalCode.astype('str')

    #Read in gejson of postal ocde area boundaries
    geoDataPath = os.path.join(get_gejson_path(), "input/postleitzahlen-deutschland.geojson")

    #Create the folium map
    fig = folium.Figure(width=500, height=500)
    map = folium.Map(location=[50.801716, 8.766453], zoom_start=11, min_zoom = 10, tiles="openstreetmap").add_to(fig)
    
    #Fill map with data
    map.choropleth(geo_data=geoDataPath,
             data=dfTripsPerPostalCode, 
             columns=['postalCode','numberOfStarts'], 
             key_on='feature.properties.plz', 
             fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='numberOfStarts')
    
    #Save map and open with browser
    filepath=os.path.abspath('data/output/PostalCodeMap.html')
    map.save(filepath)
    webbrowser.open(filepath)