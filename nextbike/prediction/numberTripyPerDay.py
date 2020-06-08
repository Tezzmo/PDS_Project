
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

    avgTripsPerDay = dfTripsPerDay['tripsPerDay'].mean()

    dfTripsPerDay['tripsLastDay'].fillna(avgTripsPerDay,inplace=True)
    dfTripsPerDay['tripsOneWeekAgo'].fillna(avgTripsPerDay,inplace=True)

    #Drop Dates and Features with a low correlation
    dfTripsPerDay['isTerm'] = dfTripsPerDay.apply(isTerm, axis=1)
    dfTripsPerDay.dropna(inplace=True)
    dfTripsPerDay.drop(['date','date_yesterday','date_oneWeekAgo'],axis=1,inplace=True)
    dfTripsPerDay.drop(['day','month'],axis=1,inplace=True)
    dfTripsPerDay.drop(['temperatureMIN','temperatureMAX'],axis=1,inplace=True)

    return dfTripsPerDay





def retrainModel_NumberOfTrips(dfTripsPerDay, optimalHyperparameterTest):

    #Mean values for compare
    meanOfAllValues = 992
    meanOfPrevoiusMonth = 1025

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

    #Compare with mean methods
    allPreviousAvg = [meanOfAllValues] * len(y_test)
    errAllPreviousAvg = mean_absolute_error(y_test, allPreviousAvg)
    err_mseAllPreviousAvg = mean_squared_error(y_test, allPreviousAvg)

    previousMonthAvg = [meanOfPrevoiusMonth] * len(y_test)
    errPreviousMonthAvg = mean_absolute_error(y_test, previousMonthAvg)
    err_msePreviousMonthAvg = mean_squared_error(y_test, previousMonthAvg)

    print("Compare to prediction by average:  ","MAE: ",errAllPreviousAvg," MSE: ", err_mseAllPreviousAvg)
    print("Compare to Avg:  ","MAE: ",errPreviousMonthAvg," MSE: ", err_msePreviousMonthAvg)



    #Visualize the Prediction
    day = []
    for i in range(len(y_test)):
        day.append(i)

    plt.plot(day, y_test,label="Number of trips")
    plt.plot(day,y_pred_test,label="Predicted number of trips")
    plt.title("Visualization of the Prediction")
    plt.xlabel("Date")
    plt.ylabel("Number of Trips")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


    #Retrain the model on all available data
  
    #Scale the Data
    sscaler = StandardScaler()
    xScaled = sscaler.fit_transform(X)
    sscalerY = StandardScaler()
    yScaled = sscalerY.fit_transform(Y.values.reshape(-1, 1))
    
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
    #Load regression
    path = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTrips.pkl")
    rfr = load(path , mmap_mode ='r')
    pathScaler = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScaler.bin")
    sscaler = load(pathScaler)
    pathScalerY = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScalerY.bin")
    sscalerY = load(pathScalerY)
  
    return rfr,sscaler, sscalerY

def predict_NumberOfTrips(dfInput, model, sscaler, sscalerY):

    #Mean values for compare
    meanOfAllValues = 992
    meanOfPrevoiusMonth = 1025

    #Create inputs dataframe and scale it
    df = dfInput.copy()
    features = createFeatures(df)
    goal = features['tripsPerDay']
    features.drop('tripsPerDay',inplace=True,axis=1,errors='ignore')
    featureValues = features.values
    xScaled = sscaler.transform(featureValues)

    #Predict and inverse transformation
    prediction = model.predict(xScaled)
    prediction = sscalerY.inverse_transform(prediction)
 
    # Save data
    path = os.path.join(utils.get_prediction_path(), "output/NumberOfTripPrediction.csv")
    df['tripsPerDay'] = prediction
    df.to_csv(path, index=False)

    print("Prediction is saved to csv --> output/NumberOfTripPrediction.csv")

    errTest = mean_absolute_error(goal, prediction)
    err_mseTest = mean_squared_error(goal, prediction)
    err_r2Test = r2_score(goal, prediction)

    print("Test  :  ","MAE: ",errTest,"MSE: ",err_mseTest,"R^2: ",err_r2Test)


    #Compare with mean methods
    allPreviousAvg = [meanOfAllValues] * len(goal)
    errAllPreviousAvg = mean_absolute_error(goal, allPreviousAvg)
    err_mseAllPreviousAvg = mean_squared_error(goal, allPreviousAvg)

    previousMonthAvg = [meanOfPrevoiusMonth] * len(goal)
    errPreviousMonthAvg = mean_absolute_error(goal, previousMonthAvg)
    err_msePreviousMonthAvg = mean_squared_error(goal, previousMonthAvg)

    print("Compare to prediction by average:  ","MAE: ",errAllPreviousAvg," MSE: ", err_mseAllPreviousAvg)
    print("Compare to Avg:  ","MAE: ",errPreviousMonthAvg," MSE: ", err_msePreviousMonthAvg)


    # Plot data
    # TODO Fix legend and axis
    plt.plot(dfInput['date'],prediction, label="Predicted number of trips")
    plt.plot(dfInput['date'],goal, label="Number of trips")
    plt.title("Visualization of the Prediction")
    plt.xlabel("Date")
    plt.ylabel("Number of Trips")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    return df


def isTerm(row):
    value = -1
    if datetime.datetime(2019, 1, 7) <= row['date'] <= datetime.datetime(2019, 2, 15):
        value = 1
    elif datetime.datetime(2019, 4, 15) <= row['date'] <= datetime.datetime(2019, 7, 19):
        value = 1
    elif datetime.datetime(2019, 10, 14) <= row['date'] <= datetime.datetime(2019, 12, 20):
        value = 1
    else:
        value = 0
    return value