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
dfTripsF[dfTripsF['weekend']==False]['weekend'] = 0
dfTripsF[dfTripsF['weekend']==True]['weekend'] = 1
dfTripsF['weekend'] = dfTripsF.weekend.astype('int64')
#%%
dfTripsF.drop('weekend',axis=1,inplace=True)
dfTripsF

#%%
dfTripsF.info()
#%%
#Build new features
#sPlaceNumber -- Stupid ?

#dfTripsF['sPlaceNumber_Pow2'] = dfTripsF['sPlaceNumber']**2
#dfTripsF['sPlaceNumber_Pow3'] = dfTripsF['sPlaceNumber']**3
#dfTripsF['sPlaceNumber_Pow4'] = dfTripsF['sPlaceNumber']**4
#dfTripsF['sPlaceNumber_Pow5'] = dfTripsF['sPlaceNumber']**5

dfTripsF['sHour2'] = dfTripsF['sHour']**2
dfTripsF['sHour3'] = dfTripsF['sHour']**3
dfTripsF['sHour4'] = dfTripsF['sHour']**4
dfTripsF['sHour5'] = dfTripsF['sHour']**5

#%%
dfTripsF.corr()


#####################################################################LINEAR REGRESSION###############################################
#%%
from sklearn.preprocessing import StandardScaler
st_scaler = StandardScaler()
dfTripsF.dropna(inplace=True)

y = dfTripsF[['duration']]
x = dfTripsF.drop('duration',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

x_train=st_scaler.fit_transform(x_train)
x_test=st_scaler.transform(x_test)
#Train model
linReg = LinearRegression().fit(x_train, y_train)

# %%
predict = linReg.predict(x_test)
mean_absolute_error(y_test, predict)

# %%
dfTripsF.describe()

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



###################################################################################TIME TEST###############################################

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










##########################################################################NEURAL NETWORK###########################################

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#%%
dfTripsF.dropna()
X = dfTripsF.drop("duration", axis=1).reset_index().drop('index',axis=1)
y = dfTripsF["duration"].tolist()
y = pd.DataFrame(y)

#%%
X = X.drop(['sPlaceNumber','temperature','sHour'],axis=1)
#%%
X.describe()
#%%
X=[
    [0,2],
    [1,2],
    [2,3],
    [2,2],
    [4,0],
    [3,5],
    [4,1],
    [2,4],
    [2,5],
    [4,1]
]
X = pd.DataFrame(X)
y=[0,2,3,4,6,7,8,9,10,11]
y = pd.DataFrame(y)

#%%
X
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
optimizer = keras.optimizers.RMSprop(0.001)

#%%
model.compile(loss='mse',
             optimizer=optimizer,
             metrics=["mae", "mse"])


#%%
model.summary()

#%%
model.predict(X_train_scaled[:10])

#%%
epochs = 2

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
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("MAE: ", mean_absolute_error(y_test, y_pred))






# %%
