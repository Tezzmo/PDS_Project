import numpy as np
import sklearn as slk
from .. import *
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Train the model with the input trip dataframe. 
# Uses randomforestregressor for maximum prediction performance and time performance.  
def train_model(df,clean_outliner = True, train_test = True, return_error = True):
    dfTrips = df.copy()
    if clean_outliner:
        dfTrips = io.input.drop_outliners(dfTrips)
        
    # Import weather data.
    dfWeather = io.getWeatherData()
    dfWeatherMinutes = pd.DataFrame({'date': pd.date_range('2019-01-01', '2020-01-01', freq='min', closed='left')})
    dfWeatherMinutes = dfWeatherMinutes.set_index('date')
    dfWeatherMinutes = dfWeatherMinutes.join(dfWeather)
    dfWeatherMinutes = dfWeatherMinutes.fillna(axis='index', method='ffill')

    # Join trips and weather data. Prepare trip data.
    dfTrips = dfTrips.join(dfWeatherMinutes, how='left', on='sTime')
    dfTrips = dfTrips.drop(columns=['eTime', 'eLong', 'eLat', 'ePlaceNumber'],errors='ignore')
    dfTrips['sMonth'] = dfTrips['sTime'].dt.month
    dfTrips['sDay'] = dfTrips['sTime'].dt.weekday
    dfTrips['sHour'] = dfTrips['sTime'].dt.hour
    dfTrips['sMinute'] = dfTrips['sTime'].dt.minute
    dfTrips['durationInSec'] = dfTrips['duration'].dt.total_seconds()
    dfTrips.drop(inplace=True, columns=['bNumber','sLong','sLat','precipitation','sMonth','sDay','bType','duration'], errors='ignore')

    # Include autocorrelation. Mean duration of the hour before the trip. 
    # If no mean is available the mean of the total dataset is used.
    dfGroups = dfTrips.resample('h',on='sTime')['durationInSec'].mean().shift(1)
    allMean = dfTrips['durationInSec'].mean()
    dfTrips['autoCorr'] = allMean
    for label,content in dfGroups.items():
        dfTrips.loc[dfTrips['sTime'].dt.floor('H') == label,['autoCorr']] = content
    dfTrips['autoCorr'].fillna(allMean,inplace=True)

    # Prediction
    dfPred = pd.concat([dfTrips, pd.get_dummies(dfTrips['sPlaceNumber'],drop_first=True,dtype='int')],axis=1)
    Y = dfPred['durationInSec'].values
    dfPred.drop(inplace=True,columns=['durationInSec'],errors='ignore')
    X = dfPred.values
    if train_test:
        x_test, x_train, y_test, y_train = train_test_split(X,Y, test_size=0.3)
    else:
        x_test, x_train, y_test, y_train = train_test_split(X,Y, test_size=0.01)
    
    sscaler = StandardScaler()
    x_train = sscaler.fit_transform(x_train)
    x_test = sscaler.transform(x_test)
    sscaler2 = StandardScaler()
    y_train = sscaler2.fit_transform(y_train.reshape(-1,1))
    y_test = sscaler2.transform(y_test.reshape(-1,1))

    model = RandomForestRegressor(n_jobs=-1,n_estimators=100)

    model.fit(x_train,y_train)
    train_prediction = sscaler2.inverse_transform(model.predict(x_train))
    test_prediction = sscaler2.inverse_transform(model.predict(x_test))
    
    y_train = sscaler2.inverse_transform(y_train)
    y_test = sscaler2.inverse_transform(y_test)

    r2_train = np.round(r2_score(y_train,train_prediction),3)
    r2_test = np.round(r2_score(y_test,test_prediction),3)
    mae_train = np.round(mean_absolute_error(y_train,train_prediction),3) 
    mae_test = np.round(mean_absolute_error(y_test,test_prediction),3)
    io.save_model(model)
    return model, r2_train, r2_test, mae_train, mae_test


def predict(y_test):
    model = io.read_model()
    
    scaler = StandardScaler()
    y_test = scaler.fit_transform(y_test)

    y_prediction = model.predict(y_test)

    return y_prediction