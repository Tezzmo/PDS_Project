import folium
import matplotlib.pyplot as plt

# visualize number of bikes per fixed station and time
def visualizeNumberOfBikesPerStationMap(pointInTime, dfStations, dfStationBikeNumber):

    # create map
    m = folium.Map(location=[50.8008, 8.7667], zoom_start=13, tiles='Stamen Toner')

    # get number of bikes for all stations at specific time
    bikesPerStation = dfStationBikeNumber.loc[pointInTime]
    
    # iterrate over all stations
    for index, row in dfStations.iloc[1:].iterrows():     

        colorStation = ''

        # get number of bikes for station
        radiusStation = int(bikesPerStation[int(row.name)])

        # set colorcode for marker based on number of bikes
        if  radiusStation < 11:
            colorStation = '#ff0000'
        elif 10 < radiusStation < 21:
            colorStation = '#e4e400'
        elif 20 < radiusStation < 31:
            colorStation = '#008000'
        elif 30 < radiusStation < 41:
            colorStation = '#1e90ff'
        elif radiusStation > 40:
            colorStation = '#0000ff'

        # configure and set marker to map
        folium.CircleMarker(
            location=[row['pLat'], row['pLong']],
            radius=radiusStation,
            popup=row['pName'],
            color=colorStation,
            fill=True,
            fill_color=colorStation
        ).add_to(m)
    
    # add an color legend to the map
    legend_html =   '''
                <div style="position: fixed; bottom: 50px; left: 50px; width: 150px; height: 130px; border:2px solid grey; z-index:9999; font-size:14px; background-color: white">
                &nbsp; <b>Color Legend</b><br>
                &nbsp; <text style='color: #ff0000;'>00 - 10 bikes/station</text><br>
                &nbsp; <text style='color: #e4e400;'>11 - 20 bikes/station</text><br>
                &nbsp; <text style='color: #008000;'>21 - 30 bikes/station</text><br>
                &nbsp; <text style='color: #1e90ff;'>31 - 40 bikes/station</text><br>
                &nbsp; <text style='color: #0000ff;'>41 - 50 bikes/station</text><br>
                </div>
                ''' 

    m.get_root().html.add_child(folium.Element(legend_html))

    #return final map
    return m

def visualizeNumberOfBikesPerStationBarplot(pointInTime, dfStations, dfStationBikeNumber):
    bikesPerStation = dfStationBikeNumber.loc[pointInTime].array
    stationNames = dfStations['pName'].iloc[1:].array

    plt.figure(figsize=(20,10))
    plt.bar(stationNames, bikesPerStation)
    plt.xticks(rotation=90)
    plt.show()
