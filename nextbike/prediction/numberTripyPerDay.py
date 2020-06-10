
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

    #Save unique dates
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

    #Get the mean for fill of nans
    avgTripsPerDay = dfTripsPerDay['tripsPerDay'].mean()

    #Fill nans with mean
    dfTripsPerDay['tripsLastDay'].fillna(avgTripsPerDay,inplace=True)
    dfTripsPerDay['tripsOneWeekAgo'].fillna(avgTripsPerDay,inplace=True)

    #Drop Dates and Features with a low correlation
    dfTripsPerDay['isTerm'] = dfTripsPerDay.apply(isTerm, axis=1)
    dfTripsPerDay.dropna(inplace=True)
    dfTripsPerDay.drop(['date','date_yesterday','date_oneWeekAgo'],axis=1,inplace=True)
    dfTripsPerDay.drop(['day','month'],axis=1,inplace=True)
    dfTripsPerDay.drop(['temperatureMIN','temperatureMAX'],axis=1,inplace=True)

    return dfTripsPerDay


# Trains a model to predict the number of trips on the next day
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

    #Scale the inputdata
    sscaler = StandardScaler()
    x_train = sscaler.fit_transform(x_train)
    x_test = sscaler.transform(x_test)

    #Sacle the output data
    sscalerY = StandardScaler()
    y_train = sscalerY.fit_transform(y_train.values.reshape(-1, 1))
    y_test = sscalerY.transform(y_test.values.reshape(-1, 1))
    #Find the optimal Hyperparameter by testing multiple combinations using RandomizedSearchCV
    if optimalHyperparameterTest == True:
        
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1500, num = 10)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}


        rf = RandomForestRegressor()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 150, cv = 3, verbose=2, random_state=45, n_jobs = -1)
        rf_random.fit(x_train,y_train)

        #Take the best values
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

    #Predict the test data and rescale them
    y_pred_test = reg.predict(x_test)
    y_pred_test = sscalerY.inverse_transform(y_pred_test)
    y_test = sscalerY.inverse_transform(y_test)

    #Evaluate the model based on test set
    errTest = mean_absolute_error(y_test, y_pred_test)
    err_mseTest = mean_squared_error(y_test, y_pred_test)
    err_r2Test = r2_score(y_test, y_pred_test)

    #Predict the train data and rescale them
    y_pred_train = reg.predict(x_train)
    y_pred_train = sscalerY.inverse_transform(y_pred_train)
    y_train = sscalerY.inverse_transform(y_train)

    #Evaluate the model based on train set
    errTrain = mean_absolute_error(y_train, y_pred_train)
    err_mseTrain = mean_squared_error(y_train, y_pred_train)
    err_r2Train = r2_score(y_train, y_pred_train)

    #Print the results
    print("Test  :  ","MAE: ",errTest,"MSE: ",err_mseTest,"R^2: ",err_r2Test)
    print("Train :  ","MAE: ",errTrain,"MSE: ",err_mseTrain,"R^2: ",err_r2Train)


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
    

#Loads the saved ml-model and scaler
def loadModel_NumberOfTrips():
    #Load random forest regressor
    path = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTrips.pkl")
    rfr = load(path , mmap_mode ='r')
    #Load scaler for input data
    pathScaler = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScaler.bin")
    sscaler = load(pathScaler)
    #Load scaler for target feature
    pathScalerY = os.path.join(utils.get_ml_path(), "numberOfTrips/randomForestRegressor_NumberOfTripsScalerY.bin")
    sscalerY = load(pathScalerY)
  
    return rfr,sscaler, sscalerY

#Predicts the number of trips for the next day
def predict_NumberOfTrips(dfInput, model, sscaler, sscalerY):

    #Mean values for compare
    meanOfAllValues = 1672
    meanOfPrevoiusMonth = 2921

    #Create inputs dataframe and scale it
    df = dfInput.copy()
    #Create necessary features
    features = createFeatures(df)
    #Save the true values to evaluate later
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

    #Evaluate the prediction
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

    print("Compare to prediction by average (last 6 months):  ","MAE: ",errAllPreviousAvg," MSE: ", err_mseAllPreviousAvg)
    print("Compare to prediction by average (last month):  ","MAE: ",errPreviousMonthAvg," MSE: ", err_msePreviousMonthAvg)


    # Plot data
    # TODO Fix legend and axis
    plt.plot(dfInput['date'],prediction, label="Predicted number of trips")
    plt.plot(dfInput['date'],goal, label="Number of trips")
    plt.title("Visualization of the Prediction")
    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Trips")
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

    return df


#Checks if uni is open
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