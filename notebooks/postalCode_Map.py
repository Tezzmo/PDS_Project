#%%
import folium
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib

#Get the numbers DRAFT!!!
!pip install e ..

#%%
import nextbike

#%%
df = nextbike.io.read_file()


# %%
import pandas as pd
import datetime
df = pd.read_csv("inputData.csv",sep=',')
#%%
dfFiltered = df[['trip','datetime','p_lat','p_lng']]
dfFiltered = dfFiltered[dfFiltered['trip']=='first']

#%%
dfFiltered['datetime'] = pd.to_datetime(dfFiltered['datetime'])
dfFiltered.info()



# %%
dfFiltered['month'] = pd.DatetimeIndex(dfFiltered["datetime"]).month
# %%
dfFilteredRdy = dfFiltered[(dfFiltered['month'] == 6)|(dfFiltered['month'] == 7)|(dfFiltered['month'] == 7)|(dfFiltered['month'] == 8)]
dfFilteredRdy

#%%
#Get postalcodes
nnc = joblib.load('nearestNeighbor_PostalCode.pkl' , mmap_mode ='r')
toClassify = dfFilteredRdy[['p_lat','p_lng']]
prediction = nnc.predict(toClassify)
dfFilteredRdy['s_postalcode'] = prediction 
dfFilteredRdy


#%%
startsPerPostalcodeAndMonth = []

for month in dfFilteredRdy['month'].unique():

    dfDataForMonth = dfFilteredRdy[dfFilteredRdy['month']==month]
    groups = dfDataForMonth.groupby('s_postalcode')

    for key in groups.groups.keys():
        postalCode = key
        numberOfStarts = len(groups.get_group(key))

        startsPerPostalcodeAndMonth.append([month,postalCode,numberOfStarts])

# %%
dfStartsPerPostalcodeAndMonth = pd.DataFrame(startsPerPostalcodeAndMonth,columns=['month','ZIP','Number'])
dfStartsPerPostalcodeAndMonth['ZIP'] = dfStartsPerPostalcodeAndMonth.ZIP.astype('str')

data1 = dfStartsPerPostalcodeAndMonth[dfStartsPerPostalcodeAndMonth['month']==8]
data1

#%%
map = folium.Map(location=[50.802578, 8.766052], default_zoom_start=10)


# %%
map.choropleth(geo_data="postleitzahlen-deutschland(1).geojson",
             data=data1, # my dataset
             columns=['ZIP','Number'], # zip code is here for matching the geojson zipcode, sales price is the column that changes the color of zipcode areas
             key_on='feature.properties.plz', # this path contains zipcodes in str type, this zipcodes should match with our ZIP CODE column
             fill_color='BuPu', fill_opacity=0.7, line_opacity=0.2,
             legend_name='Number')

# %%
map 


# %%
map = None

# %%
from IPython.display import HTML

# %%
display(map)  