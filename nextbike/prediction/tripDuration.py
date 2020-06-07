import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals.joblib import dump, load
from . import utils
import os

def createFeatures(dfTrips, dfWeather):

    # Transform weather into minute resolution
    dfWeatherMinutes = pd.DataFrame({'date': pd.date_range('2019-01-01', 
    '2020-01-01', freq='min', closed='left')})
    dfWeatherMinutes = dfWeatherMinutes.set_index('date')
    dfWeatherMinutes = dfWeatherMinutes.join(dfWeather)
    dfWeatherMinutes = dfWeatherMinutes.fillna(axis='index', method='ffill')

    # Merge weather into trips
    
    dfTrips = dfTrips.join(dfWeatherMinutes, how='left', on='sTime')
    
    
    # Create time features
    dfTrips['sMonth'] = dfTrips['sTime'].dt.month
    dfTrips['sDay'] = dfTrips['sTime'].dt.weekday
    dfTrips['sHour'] = dfTrips['sTime'].dt.hour
    dfTrips['sMinute'] = dfTrips['sTime'].dt.minute


    dfTrips.drop(axis=1,labels=['eTime', 'eLong', 'eLat', 'ePlaceNumber'],errors='ignore',inplace=True)
    # Create autoCorr feature
    dfGroups = dfTrips.resample('h',on='sTime')['durationInSec'].mean().shift(1)
    allMean = dfTrips['durationInSec'].mean()
    dfTrips['autoCorr'] = allMean
    for label,content in dfGroups.items():
        dfTrips.loc[dfTrips['sTime'].dt.floor('H') == label,['autoCorr']] = content
    dfTrips['autoCorr'].fillna(allMean,inplace=True)
    dfTrips.dropna(inplace=True)

    dfTrips.drop(inplace=True, columns=['bNumber','sTime','sLong','sLat',
    'precipitation','sMonth','sDay','bType','sPostalCode','ePostalCode','duration'], errors='ignore')
    

    return dfTrips



def retrainModel_DurationOfTrips(dfTrips,dfWeather, optimalHyperparameterTest):
    
    dfTrips = createFeatures(dfTrips,dfWeather)

    # Create train test split
    Y = dfTrips['durationInSec']
    X = dfTrips.drop('durationInSec',axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

    # Scale the data
    sscaler = StandardScaler()
    x_train = sscaler.fit_transform(x_train)
    x_test = sscaler.transform(x_test)

    sscalerY = StandardScaler()
    y_train = sscalerY.fit_transform(y_train.values.reshape(-1, 1))
    y_test = sscalerY.transform(y_test.values.reshape(-1, 1))

    ###Find the optimal Hyperparameter by testing multiple (150) combinations using RandomizedSearchCV
    if optimalHyperparameterTest == True:
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1500, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}


        # Use the random grid to search for best hyperparameters
        # First create the base model to tune
        rf = RandomForestRegressor()
        # Random search of parameters, using 3 fold cross validation, 
        # search across 150 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
         n_iter = 50, cv = 3, verbose=2, random_state=45, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(x_train,y_train)

        reg = RandomForestRegressor(n_jobs=-1,
                            n_estimators = rf_random.best_params_['n_estimators'],
                            min_samples_split = rf_random.best_params_['min_samples_split'],
                            min_samples_leaf = rf_random.best_params_['min_samples_leaf'],
                            max_features = rf_random.best_params_['max_features'],
                            max_depth = rf_random.best_params_['max_depth'],
                            bootstrap = rf_random.best_params_['bootstrap']
                            )

    else:
        #Else use default
        reg = RandomForestRegressor(n_jobs=-1,n_estimators=1000)


    #Train the model
    reg.fit(x_train,y_train)

    #Evaluate the model based on the train test split
    y_pred_test = reg.predict(x_test)
    y_pred_test = sscalerY.inverse_transform(y_pred_test)
    y_test = sscalerY.inverse_transform(y_test)

    errTest = mean_absolute_error(y_test, y_pred_test)
    err_mseTest = mean_squared_error(y_test, y_pred_test)
    err_r2Test = r2_score(y_test, y_pred_test)

    y_pred_train = reg.predict(x_train)

    y_pred_train = sscalerY.inverse_transform(y_pred_train)
    y_train = sscalerY.inverse_transform(y_train)

    errTrain = mean_absolute_error(y_train, y_pred_train)
    err_mseTrain = mean_squared_error(y_train, y_pred_train)
    err_r2Train = r2_score(y_train, y_pred_train)

    print("Test  :  ","MAE: ",errTest,"MSE: ",err_mseTest,"R^2: ",err_r2Test)
    print("Train :  ","MAE: ",errTrain,"MSE: ",err_mseTrain,"R^2: ",err_r2Train)


    # Visualize TODO

    #Save the model
    path = os.path.join(utils.get_ml_path(), "DurationOfTrips/randomForestRegressor_DurationOfTrips.pkl")
    dump(reg,path)

    # Save the scaler
    pathScaler = os.path.join(utils.get_ml_path(), "DurationOfTrips/randomForestRegressor_DurationOfTripsScaler.bin")
    dump(sscaler,pathScaler)

    pathScalerY = os.path.join(utils.get_ml_path(), "DurationOfTrips/randomForestRegressor_DurationOfTripsScalerY.bin")
    dump(sscalerY,pathScalerY)



def loadModel_DurationOfTrips():
    # Load regression
    path = os.path.join(utils.get_ml_path(), "DurationOfTrips/randomForestRegressor_DurationOfTrips.pkl")
    rfr = load(path , mmap_mode ='r')
    pathScaler = os.path.join(utils.get_ml_path(), "DurationOfTrips/randomForestRegressor_DurationOfTripsScaler.bin")
    sscaler = load(pathScaler)
    pathScalerY = os.path.join(utils.get_ml_path(), "DurationOfTrips/randomForestRegressor_DurationOfTripsScalerY.bin")
    sscalerY = load(pathScalerY)

    return rfr,sscaler, sscalerY

    



def predict_DurationOfTrips(dfInput,dfWeather, model, sscaler, sscalerY):
    dfTrips = dfInput.copy()

    
    
    #Create inputs dataframe and scale it
    features = createFeatures(dfTrips,dfWeather)
    features.drop('durationInSec',inplace=True,axis=1,errors='ignore')
    featureValues = features.values
    xScaled = sscaler.transform(featureValues)

    #Predict and inverse transformation
    prediction = model.predict(xScaled)
    prediction = sscalerY.inverse_transform(prediction)
    
    
    # Save data
    path = os.path.join(utils.get_prediction_path(), "output/DurationOfTripsPrediction.csv")
    features['Duration of trips'] = prediction
    features.to_csv(path, index=False)

    # Plot data
    # TODO Fix legend and axis
    print('Prediction done and saved to csv --> "output/DurationOfTripsPrediction.csv"')
    return dfTrips