#%%
!pip install -e ..

# %%
import nextbike
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# %%
dfTrips = nextbike.io.readFinalTrips()
dfTripsC=dfTrips
dfTripsC.info()



#%%
#Get Weather
dfWeather = nextbike.io.getWeatherData()
dfWeatherMinutes = pd.DataFrame({'date': pd.date_range('2019-01-01', '2020-01-01', freq='min', closed='left')})
dfWeatherMinutes = dfWeatherMinutes.set_index('date')
dfWeatherMinutes = dfWeatherMinutes.join(dfWeather)
dfWeatherMinutes = dfWeatherMinutes.fillna(axis='index', method='ffill')
dfWeatherMinutes

dfWeatherMinutes['sTime'] = dfWeatherMinutes.index

#%%
#Rundet zur nächsten glatten 10 min ab und merge mit wetter daten
#dfTripsC['sTimeRoundet'] = dfTripsC['sTime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,10*(dt.minute // 10)))
dfTripsF = dfTripsC.join(dfWeatherMinutes,how='inner',lsuffix='_l', rsuffix='_r', on='sTime' )

dfTripsF['sYear'] = dfTripsF['sTime'].dt.year
dfTripsF['sMonth'] = dfTripsF['sTime'].dt.month
dfTripsF['sDay'] = dfTripsF['sTime'].dt.day
dfTripsF['sHour'] = dfTripsF['sTime'].dt.hour
dfTripsF['sMinute'] = dfTripsF['sTime'].dt.minute
dfTripsF['sDayOfWeek'] = dfTripsF['sTime'].dt.weekday

dfTripsF.drop(['sTime_l','sTime_r'],axis=1,inplace=True)

dfTripsF['duration'] = dfTripsF['duration'].dt.total_seconds()

dfTripsF.drop(['eTime','eLong','eLat','ePlaceNumber'],axis=1,inplace=True)

#dfTripsF.loc[dfTripsF['precipitation'] >= 0.1, 'isRain'] = 1
#dfTripsF.loc[dfTripsF['precipitation'] < 0.1, 'isRain'] = 0

dfTripsF.drop(['bNumber','sLong','sLat'],axis=1,inplace=True)
dfTripsF.drop(['sDay','sYear','sMinute','bType'],axis=1,inplace=True)

dfTripsF.drop('sTime',axis=1,inplace=True)
dfTripsF.drop('precipitation',axis=1,inplace=True)

#%%
dfTripsF

#%%
dfTripsF[dfTripsF['weekend']==False]['weekend'] = 0
dfTripsF[dfTripsF['weekend']==True]['weekend'] = 1
dfTripsF['weekend'] = dfTripsF.weekend.astype('int64')
#%%
dfTripsF.drop('weekend',axis=1,inplace=True)
dfTripsF



#################   Feature Creation
#%%


#%%
dfTripsF.info()


##########################################################################NEURAL NETWORK###########################################

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#%%
dfTripsF.dropna(inplace=True)
X = dfTripsF.drop("duration", axis=1).reset_index().drop('index',axis=1)
y = dfTripsF["duration"].tolist()
y = pd.DataFrame(y)


#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%%
X_train.describe()
#%%
from sklearn.preprocessing import StandardScaler
#%%
st_scaler = StandardScaler()
st_scaler.fit(X_train)
X_train_scaled = st_scaler.transform(X_train)

#%%
model = keras.Sequential(
    [layers.Dense(36, activation="relu", input_shape=[X_train.shape[1]]),
    layers.Dense(36, activation="relu"),
     layers.Dense(1)])

#%%
optimizer = keras.optimizers.RMSprop(0.0005)

#%%
model.compile(loss='mse',
             optimizer=optimizer,
             metrics=["mae", "mse"])


#%%
model.summary()

#%%
epochs = 8

history = model.fit(X_train_scaled, y_train.values,
                   epochs=epochs, validation_split=0.2)

#%%
history_df = pd.DataFrame(history.history)
history_df

#%%
X_test_scaled = st_scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)
y_pred
#%%
print("MAE: ", mean_absolute_error(y_test, y_pred))

#%%
dfTripsF.corr()

# %%
