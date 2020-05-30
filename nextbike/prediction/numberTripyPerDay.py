import nextbike
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


    ###Feature creation
    #Get the date of yesterday and one week ago
    dfTripsPerDayFilterd['date_yesterday'] = pd.DatetimeIndex(dfTripsPerDayFilterd['date']) - pd.DateOffset(1)
    dfTripsPerDayFilterd['date_oneWeekAgo'] = pd.DatetimeIndex(dfTripsPerDayFilterd['date']) - pd.DateOffset(7)

    #Get the number of trips for this dates
    tripsLastDay = []
    tripsOneWeekAgo = []

    lastDay = list(dfTripsPerDayFilterd['date_yesterday'].unique())
    oneWeekEarlier = list(dfTripsPerDayFilterd['date_oneWeekAgo'].unique())

    #Try/except is necessary, because of the outlier drop 
    for date in lastDay:
        try:
            valueDay = dfTripsPerDayFilterd[dfTripsPerDayFilterd['date'] == date]['tripsPerDay'].values[0]
        except:
            valueDay = None

        tripsLastDay.append(valueDay)


    for date in oneWeekEarlier:
        try:
            valueWeek = dfTripsPerDayFilterd[dfTripsPerDayFilterd['date'] == date]['tripsPerDay'].values[0]
        except:
            valueWeek = None

        tripsOneWeekAgo.append(valueWeek)

    
    #Fill in created features
    dfTripsPerDayFilterd['tripsLastDay'] = tripsLastDay
    dfTripsPerDayFilterd['tripsOneWeekAgo'] = tripsOneWeekAgo

    #Drop Dates and Features with a low correlation
    dfTripsPerDayFilterd.dropna(inplace=True)
    dfTripsPerDayFilterd.drop(['date','date_yesterday','date_oneWeekAgo'],axis=1,inplace=True)
    dfTripsPerDayFilterd.drop(['day','month'],axis=1,inplace=True)


    #Split into Week and Weekend
    dfTripsPerDayWeek = dfTripsPerDayFilterd[(dfTripsPerDayFilterd['dayOfWeek']==0)|(dfTripsPerDayFilterd['dayOfWeek']==1)|(dfTripsPerDayFilterd['dayOfWeek']==2)|(dfTripsPerDayFilterd['dayOfWeek']==3)|(dfTripsPerDayFilterd['dayOfWeek']==4)|(dfTripsPerDayFilterd['dayOfWeek']==5)]
    dfTripsPerDayWeekend = dfTripsPerDayFilterd[(dfTripsPerDayFilterd['dayOfWeek']==5)|(dfTripsPerDayFilterd['dayOfWeek']==6)]

    YWeek= dfTripsPerDayWeek['tripsPerDay']
    XWeek = dfTripsPerDayWeek.drop('tripsPerDay',axis=1).values

    YWeekEnd= dfTripsPerDayWeekend['tripsPerDay']
    XWeekEnd = dfTripsPerDayWeekend.drop('tripsPerDay',axis=1).values


    ################################################################### Choose 
    #Make a train test split
    Y = dfTripsPerDayFilterd['tripsPerDay']
    X = dfTripsPerDayFilterd.drop('tripsPerDay',axis=1).values
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

    #Scale the Data
    sscaler = StandardScaler()
    x_train = sscaler.fit_transform(x_train)
    x_test = sscaler.transform(x_test)

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
    err = mean_absolute_error(y_test, y_pred_test)
    err_mse = mean_squared_error(y_test, y_pred_test)
    err_r2 = r2_score(y_test, y_pred_test)

    print("MSE: ",err,"MAE: ",err_mse,"R^2: ",err_r2)

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
    reg.fit(xScaled,Y)

    return reg,sscaler

def loadModel_NumberOfTrips():
    pass

def predict_NumberOfTrips(df, regressor, sscaler):

    dfSalced = sscaler.transform(df)
    prediction = regressor.predict(dfSalced)

    return prediction
