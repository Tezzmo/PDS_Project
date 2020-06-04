
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

def createFeatures(dfTripsPerDay):

    ###Feature creation
    #Get the date of yesterday and one week ago
    dfTripsPerDay['date_yesterday'] = pd.DatetimeIndex(dfTripsPerDay['date']) - pd.DateOffset(1)
    dfTripsPerDay['date_oneWeekAgo'] = pd.DatetimeIndex(dfTripsPerDay['date']) - pd.DateOffset(7)

    #Get the number of trips for this dates
    tripsLastDay = []
    tripsOneWeekAgo = []

    lastDay = list(dfTripsPerDay['date_yesterday'].unique())
    oneWeekEarlier = list(dfTripsPerDay['date_oneWeekAgo'].unique())

    #Try/except is necessary, because of the outlier drop 
    for date in lastDay:
        try:
            valueDay = dfTripsPerDay[dfTripsPerDay['date'] == date]['tripsPerDay'].values[0]
        except:
            valueDay = None

        tripsLastDay.append(valueDay)


    for date in oneWeekEarlier:
        try:
            valueWeek = dfTripsPerDay[dfTripsPerDay['date'] == date]['tripsPerDay'].values[0]
        except:
            valueWeek = None

        tripsOneWeekAgo.append(valueWeek)

    
    #Fill in created features
    dfTripsPerDay['tripsLastDay'] = tripsLastDay
    dfTripsPerDay['tripsOneWeekAgo'] = tripsOneWeekAgo

    #Drop Dates and Features with a low correlation
    dfTripsPerDay.dropna(inplace=True)
    dfTripsPerDay.drop(['date','date_yesterday','date_oneWeekAgo'],axis=1,inplace=True)
    dfTripsPerDay.drop(['day','month'],axis=1,inplace=True)

    return dfTripsPerDay





def retrainModel_NumberOfTrips(dfTripsPerDay, optimalHyperparameterTest):

    ###First filter for outliers
    #Get standart deviation of trips per day
    std = dfTripsPerDay['tripsPerDay'].std()

    #For each month
    for i in range(1,13):
        #Get the average number of trips per month
        df = dfTripsPerDay[dfTripsPerDay['month']==i]
        avg = df['tripsPerDay'].mean()

        #Only use days where the average is within the standartdeviation
        row = df[(df['tripsPerDay'] <= avg+std)&(df['tripsPerDay'] >= avg-std)]

        #Combine the data for the months again
        if i == 1:
            dfTripsPerDayFilterd = row
        else:
            dfTripsPerDayFilterd = pd.concat([dfTripsPerDayFilterd,row])

    #Create features
    dfTripsPerDayFilterd = createFeatures(dfTripsPerDayFilterd)

    #Make a train test split
    Y = dfTripsPerDayFilterd['tripsPerDay']
    X = dfTripsPerDayFilterd.drop('tripsPerDay',axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

    #Scale the Data
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
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 150, cv = 3, verbose=2, random_state=45, n_jobs = -1)
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

    #Visualize the Prediction
    day = []
    for i in range(len(y_test)):
        day.append(i)

    plt.plot(day, y_test)
    plt.plot(day,y_pred_test)
    plt.title = "Visualization of the Prediction"
    plt.xlabel = "Day"
    plt.ylabel = "Number of Trips"

    plt.show()


    #Retrain the model on all available data
  
    #Scale the Data
    sscaler = StandardScaler()
    xScaled = sscaler.fit_transform(X)
    sscalerY = StandardScaler()
    yScaled = sscalerY.fit_transform(X)
    
    reg.fit(xScaled,yScaled)
    #Save the model
    path = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTrips.pkl")
    dump(reg,path)

    # Save the scaler
    pathScaler = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScaler.bin")
    dump(sscaler,pathScaler)

    pathScalerY = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScalerY.bin")
    dump(sscalerY,pathScalerY)
    

def loadModel_NumberOfTrips():
    #Load NN-Classifier
    path = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTrips.pkl")
    rfr = load(path , mmap_mode ='r')
    pathScaler = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScaler.bin")
    sscaler = load(pathScaler)
    pathScalerY = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScalerY.bin")
    sscalerY = load(pathScalerY)
    '''
    #Visualize the Prediction
    day = []
    for i in range(len(prediction)):
        day.append(i)
  
    plt.plot(day,prediction, label="Predicted number of trips")
    plt.title = "Visualization of the Prediction"
    plt.xlabel("Datapoints of testset")
    plt.ylabel("Number of Trips")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()
    '''
    return rfr,sscaler, sscalerY

def predict_NumberOfTrips(dfInput, model, sscaler, sscalerY):

    #Create inputs dataframe and scale it
    df = dfInput.copy()
    features = createFeatures(df)
    features.drop('tripsPerDay',inplace=True,axis=1,errors='ignore')
    featureValues = features.values
    xScaled = sscaler.transform(featureValues)

    #Predict and inverse transformation
    prediction = model.predict(xScaled)
    prediction = sscalerY.inverse_transform(prediction)
    prediction = prediction.reshape(-1, 1)
    print(len(prediction))
    print(len(features))
    print(len(df))
    # Save data
    path = os.path.join(utils.get_prediction_path(), "output/NumberOfTripPrediction.csv")
    features['tripsPerDay'] = prediction
    features.to_csv(path, index=False)

    # Plot data
    # TODO Fix legend and axis
    prediction.to_csv("prediction.csv")
    plt.plot(range(0,len(prediction)),prediction, label="Predicted number of trips")
    plt.title = "Visualization of the Prediction"
    plt.xlabel("Datapoints of testset")
    plt.ylabel("Number of Trips")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    return df
