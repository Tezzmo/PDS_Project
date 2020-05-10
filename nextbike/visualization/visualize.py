import folium

def visualizeNumberOfBikesPerStation(pointInTime, dfStations, dfStationBikeNumber):

    m = folium.Map(location=[50.8008, 8.7667], zoom_start=13, tiles='Stamen Toner')

    bikesPerStation = dfStationBikeNumber.loc[pointInTime]
    for index, row in dfStations.iloc[1:].iterrows():     
        folium.CircleMarker(
            location=[row['pLat'], row['pLong']],
            radius=int(bikesPerStation[int(row.name)]),
            popup=row['pName'],
            color='#3186cc',
            fill=True,
            fill_color='#3186cc'
        ).add_to(m)

    return m