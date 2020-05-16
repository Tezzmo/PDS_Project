#%%
!pip install -e ..

# %%
import nextbike

# %%
df = nextbike.io.read_file()

#%%
dfTrips = nextbike.io.createTrips(df)

#%%
dfTrips.to_csv("savedTrips.csv",sep=';')
#%%
dfTrips = pd.read_csv("savedTrips.csv",sep=';')
#%%
dfTripsC=dfTrips
dfTripsC.info()

#%%
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



#%%
#Get Weather
dfWeather = nextbike.io.getWeatherData()
dfWeatherMinutes = pd.DataFrame({'date': pd.date_range('2019-01-01', '2020-01-01', freq='min', closed='left')})
dfWeatherMinutes = dfWeatherMinutes.set_index('date')
dfWeatherMinutes = dfWeatherMinutes.join(dfWeather)
dfWeatherMinutes = dfWeatherMinutes.fillna(axis='index', method='ffill')

#%%
dfWeatherMinutes.columns

#%%
#Rundet zur nächsten glatten 10 min ab und merge mit wetter daten
#dfTripsC['sTimeRoundet'] = dfTripsC['sTime'].apply(lambda dt: datetime.datetime(dt.year, dt.month, dt.day, dt.hour,10*(dt.minute // 10)))
dfTripsF = dfTripsC.set_index('sTime').join(dfWeatherMinutes,how='inner',lsuffix='_l', rsuffix='_r')

#%%
dfTripsF.columns
dfTripsF = dfTripsF.drop(['Unnamed: 0'],axis=1)
dfTripsF
# %%
#Change Datetime to numeric -- needed?
dfTripsF['sTime'] = pd.to_numeric(dfTripsF['sTime'])
dfTripsF['eTime'] = pd.to_numeric(dfTripsF['eTime'])


#%%
#Set duration to numeric - LÖittle higher accuracy with secounds
#dfTripsF['duration'] = round(dfTripsF['duration'].dt.total_seconds().divide(60),0)
dfTripsF['duration'] = dfTripsF['duration'].dt.total_seconds()


#%%
#Lets drop a little bit features
#Must be dropped
dfTripsF.drop(['eTime','eLong','eLat','ePlaceNumber'],axis=1,inplace=True)
#Make sno difference
dfTripsF.drop(['bNumber','sLong','sLat'],axis=1,inplace=True)


#%%
#Build new features
#sPlaceNumber -- Stupid ?

#dfTripsF['sPlaceNumber_Pow2'] = dfTripsF['sPlaceNumber']**2
#dfTripsF['sPlaceNumber_Pow3'] = dfTripsF['sPlaceNumber']**3
#dfTripsF['sPlaceNumber_Pow4'] = dfTripsF['sPlaceNumber']**4
#dfTripsF['sPlaceNumber_Pow5'] = dfTripsF['sPlaceNumber']**5

dfTripsF['sTime_Pow2'] = dfTripsF['sPlaceNumber']**2
dfTripsF['sTime_Pow3'] = dfTripsF['sPlaceNumber']**3
dfTripsF['sTime_Pow4'] = dfTripsF['sPlaceNumber']**4
dfTripsF['sTime_Pow5'] = dfTripsF['sPlaceNumber']**5

#%%
dfTripsF.dropna(inplace=True)

y = dfTripsF[['duration']]
x = dfTripsF.drop('duration',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#Train model
linReg = LinearRegression(normalize=True).fit(x_train, y_train)

# %%
predict = linReg.predict(x_test)
mean_absolute_error(y_test, predict)

# %%
linReg.coef_

# %%
y_test

# %%
dfTripsF.corr()

# %%
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from numba import jit, cuda 

#%%
@jit
def train():
    regr = make_pipeline(StandardScaler(), SVR(C=0.2, epsilon=0.2))
    regr.fit(x_train, y_train)

train()

# %%
regr.predict(x_test)

# %%
x_train





#%%
#from numba import jit, cuda 
import numpy as np 
from timeit import default_timer as timer 


# %%
n = 10000000                            
a = np.ones(n, dtype = np.float64) 
start = timer() 
nextbike.io.func2(a)
print("with GPU:", timer()-start) 

# %%
n = 10000000                            
a = np.ones(n, dtype = np.float64) 
start = timer() 
nextbike.io.func1(a)
print("no GPU:", timer()-start) 













# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# %%
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


mode = build_model()

print(model.summary())
