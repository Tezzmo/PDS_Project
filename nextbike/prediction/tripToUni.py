from .. import io
from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import sklearn.metrics
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import pickle
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
    
    return dfTrips



def trainKNNRegression(df, dfWeather ):
    
    dfTrips = createFeatures(df,dfWeather)
    
    # create new features
    dfTrips['tripToUniversity'] = dfTrips.apply(isStationAtUniversity, axis=1)
    dfTrips['isTerm'] = dfTrips.apply(isTerm, axis=1)
    dfTrips['month'] = dfTrips['sTime'].dt.month
    dfTrips['dayOfWeek'] = dfTrips['sTime'].dt.dayofweek
    dfTrips['hour'] = dfTrips['sTime'].dt.hour
    dfTrips['minute'] = dfTrips['sTime'].dt.minute
    dfTrips['isUniOpen'] = dfTrips.apply(isUniOpen, axis=1)
    
    # define input feature
    X = pd.DataFrame()
    X['sPlaceNumber'] = dfTrips['sPlaceNumber']
    X['day'] = dfTrips['dayOfWeek']
    X['isTerm'] = dfTrips['isTerm']
    X['isUniOpen'] = dfTrips['isUniOpen']
    X['month'] = dfTrips['month']
    X['hour'] = dfTrips['hour']
    X['temperature'] = dfTrips['temperature']
    X['precipitation'] = dfTrips['precipitation']
    # create dummies for sPlaceNumber
    sPlace = pd.get_dummies(dfTrips["sPlaceNumber"], drop_first=True)
    X.drop('sPlaceNumber', axis=1, inplace=True)
    X = pd.concat([X, sPlace], axis=1)

    # define target feature
    y = pd.DataFrame(dfTrips['tripToUniversity'])

    # split data in test and training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    st_scaler = StandardScaler()
    st_scaler.fit(X_train)
    X_train_scaled = st_scaler.transform(X_train)
    X_test_scaled = st_scaler.transform(X_test)
    pathScaler = os.path.join(utils.get_ml_path(), "tripsToUni/scaler.pkl")
    dump(st_scaler,pathScaler)

    #make an instance of the Model which explains 99% of Variance
    pca = PCA(0.99)
    pca.fit(X_train_scaled)
    X_train_scaled = pca.transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)
    pathPCA = os.path.join(utils.get_ml_path(), "tripsToUni/pca.pkl")
    dump(pca,pathPCA)

    knn = KNeighborsRegressor()
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    pathKNN = os.path.join(utils.get_ml_path(), "tripsToUni/knn.pkl")
    dump(knn,pathKNN)
    
    # round predicted values and return classification_report
    y_pred_rounded = np.round(y_pred, 0)

    print(classification_report(y_true=y_test, y_pred=y_pred_rounded))


def predictTripDirection(df,dfWeather):

    dfTrips = createFeatures(df,dfWeather)
    X_pred = dfTrips.copy()

    # create new features
    X_pred['tripToUniversity'] = X_pred.apply(isStationAtUniversity, axis=1)
    X_pred['isTerm'] = X_pred.apply(isTerm, axis=1)
    X_pred['month'] = X_pred['sTime'].dt.month
    X_pred['dayOfWeek'] = X_pred['sTime'].dt.dayofweek
    X_pred['hour'] = X_pred['sTime'].dt.hour
    X_pred['minute'] = X_pred['sTime'].dt.minute
    X_pred['isUniOpen'] = X_pred.apply(isUniOpen, axis=1)

    stationNumbers = [5173,5159,5178,5150,5156,5158,5153,5162,5166,5168,5155,5175,0,5165,5171,5176,5145,5163,5164,5147,5167,5157,5177,5169,5152,5160,5154,5174,5172,5144,5151,5143,5146,5141,5142,5140]
    dfStationNumbers = pd.DataFrame(stationNumbers)
    dfStationNumbers.rename(columns={0:'sPlaceNumber'}, inplace=True)
    dfStationNumbers

    # define input feature
    X = pd.DataFrame()
    X['sPlaceNumber'] = X_pred['sPlaceNumber']
    X['day'] = X_pred['dayOfWeek']
    X['isTerm'] = X_pred['isTerm']
    X['isUniOpen'] = X_pred['isUniOpen']
    X['month'] = X_pred['month']
    X['hour'] = X_pred['hour']
    X['temperature'] = X_pred['temperature']
    X['precipitation'] = X_pred['precipitation']
    X = X.join((dfStationNumbers.set_index('sPlaceNumber')), on='sPlaceNumber', how='outer')

    # create dummies for sPlaceNumber
    sPlace = pd.get_dummies(X["sPlaceNumber"], drop_first=True)
    X.drop('sPlaceNumber', axis=1, inplace=True)
    X = pd.concat([X, sPlace], axis=1)
    X.dropna(inplace=True)

    try:
        pathScaler = os.path.join(utils.get_ml_path(), "tripsToUni/scaler.pkl")
        st_scaler = load(pathScaler)

    except FileNotFoundError:
        return "Standard Scaler not found. Please train a model first."

    try:
        pathPCA = os.path.join(utils.get_ml_path(), "tripsToUni/PCA.pkl")
        pca = load(pathPCA)
    except FileNotFoundError:
        return "Principal Component Analysis not found. Please train a model first."

    try:
        pathKNN = os.path.join(utils.get_ml_path(), "tripsToUni/KNN.pkl")
        knn = load(pathKNN)
    except FileNotFoundError:
        return "K-Nearest Neighbor not found. Please train a model first."
    
    X_scaled = st_scaler.transform(X)
    X_scaled = pca.transform(X_scaled)
    y_pred = knn.predict(X_scaled)

    return y_pred

def isStationAtUniversity(row):
    # Anatomie
    if row['ePlaceNumber'] == 5140:
        val = 1
    # Neue Universitaetsbibliothek
    elif row['ePlaceNumber'] == 5145:
        val = 1
    # Am Plan/Wirtschaftswissenschaften
    elif row['ePlaceNumber'] == 5152:
        val = 1
    # Biegenstrasse/Volkshochschule
    elif row['ePlaceNumber'] == 5156:
        val = 1
    # Frankfurter Strasse/Psychologie
    elif row['ePlaceNumber'] == 5159:
        val = 1
    # Audimax
    elif row['ePlaceNumber'] == 5166:
        val = 1
    # Marbacher Weg/Pharmazie
    elif row['ePlaceNumber'] == 5167:
        val = 1
    # Philosophische Fakultaet
    elif row['ePlaceNumber'] == 5169:
        val = 1
    # Universitaetsstadion
    elif row['ePlaceNumber'] == 5175:
        val = 1
    # Universitaetsstrasse/Bibliothek Jura
    elif row['ePlaceNumber'] == 5176:
        val = 1
    # Carolinenhaus/Elisabethkirche
    elif row['ePlaceNumber'] == 5144:
        val = 1
    # Elisabeth-Blochmann-Platz
    elif row['ePlaceNumber'] == 5158:
        val = 1
    # Am Schülerpark
    elif row['ePlaceNumber'] == 5147:
        val = 1
    # Aquamar
    elif row['ePlaceNumber'] == 5171:
        val = 1
    # all other stations
    else:
        val = 0
    return val

def isTerm(row):
    value = -1
    if datetime.datetime(2019, 1, 7) <= row['sTime'] <= datetime.datetime(2019, 2, 15):
        value = 1
    elif datetime.datetime(2019, 4, 15) <= row['sTime'] <= datetime.datetime(2019, 7, 19):
        value = 1
    elif datetime.datetime(2019, 10, 14) <= row['sTime'] <= datetime.datetime(2019, 12, 20):
        value = 1
    else:
        value = 0
    return value

def isUniOpen(row):
    value = -1
    if row['dayOfWeek'] <= 5:
        if 8 <= row['hour'] <= 18:
            value = 1
        else:
            value = 0
    else:
        value = 0
    return value