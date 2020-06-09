import click
import datetime
from . import io
from . import postalCodes
from . import visualization
from . import prediction
import warnings
from IPython.display import display, Image
from IPython.core.display import HTML

@click.command()
@click.option('--train',default=None, help="Input a filename. Train the model on this data.")
@click.option('--transform',default=None,help='Input a filename. Transform this data into Trips')
@click.option('--predict', default=None,help='Input a filename. Predict on the dataset. Before prediction you need to train on a dataset.')


#Entering point
def main(train,transform,predict):
    print(train,transform,predict)

    if transform != None:
        if (train != None) & (transform != None) & (predict != None):
            dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild(train)
        else:
            dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild(transform)

    if train != None:
        if transform == None:
            dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = useExisting()
        print('This can take a few minutes')
        prediction.retrainModel_DurationOfTrips(dfTrips,dfWeather,False)

    if predict != None:
        if (train != None) & (transform != None) & (predict != None):
            print('This can take a few minutes')
            dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild(transform)

        print('This can take a few minutes')
        model, sscaler, sscalerY = prediction.loadModel_DurationOfTrips()
        prediction.predict_DurationOfTrips(dfTrips, dfWeather, model, sscaler,sscalerY)
    
    #Don't show warnings in the console
    if (train == None) & (transform == None) & (predict == None):
        warnings.filterwarnings("ignore")
        dataLoadMenue()
    else:
        print('Program finished')



#Give user the possiblity to chose input data
def dataLoadMenue():
    print("Welcome")
    print("Your options: \n1 - Use default data to create all Dataframes \n2 - Use new data to create all Dataframes \n3 - Use saved Dataframes")
    userInteraction = input("Press 1, 2 or 3 \n") 

    #Rebuild all dataframes on the default data
    if userInteraction == '1':
        print('Start to rebuild the model, this will take several minutes')
        dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild()
        #Parameter true -> enable visualizations
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,True)
    
    #Rebuild all dataframes based on a custom file 
    elif userInteraction == '2':
        print('Input the filename in data/input/')
        fileInput = input('Exp.: data.csv \n')
        dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild(fileInput)
        #Parameter false -> disable visualizations
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,False)
    
    #Load the last builded dataframes
    elif userInteraction == '3':
        print('Start to load all data')
        dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = useExisting()

        #If dfTrips == 563880 -> default dataset  -> enable visualization
        if len(dfTrips) == 563880:
            mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,True)
        else:
            mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,False)
    

#Gives user the possiblity to chose between prediction and visualization
def mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData):

    #If the dataframes were builded on the default data enable visualization
    if defaultData == True:
        print("Your options: \n1 - Visualize \n2 - Predict \n3 - Go Back \n4 - End")
        userInteraction = input("Press 1,2,3 or 4 \n") 

        #Visualization
        if userInteraction == '1':
            menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)

        #Prediction
        elif userInteraction == '2':
            menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)

        #Go back to the dataload menue
        elif userInteraction == '3':
            dataLoadMenue()

        #Terminate the ui
        elif userInteraction == '4':
            pass

        else:
            print("Invalied Input")
            mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)

    # Else disable visualization to avoid broken graphics
    #Only predictions allowed
    else:
        print("Your options: \n1 - Predict \n2 - Go back \n3 - End")
        userInteraction = input("Press 1 or 2 \n")

        if userInteraction == '1':
            menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)

        elif userInteraction == '2':
            dataLoadMenue()

        elif userInteraction == '3':
            pass

        else:
            print("Invalied Input")
            mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)


# Gives user the possiblity to chose between different visualization themes
def menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData):

    print("What are you interested in? \n 1 - Trip duration \n 2 - Number of Trips \n 3 - Start/End point of Trips \n 4 - Bikes per Station \n 5 - Weather data \n 6 - Heat map \n 7 - Go back")
    userInteraction = input("Choose a number <1 - 7> \n")
    userInteraction = int(userInteraction)

    #Chose a visualization
    if userInteraction in [1,2,3,4,5,6]:
        visualize(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,userInteraction,defaultData)
    #Go back
    elif userInteraction == 7:
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)
    #Invalid input
    else:
        print("Invalid input")
        menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)
        

# Gives user the possiblity to chose between different predictions topics
def menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData):

    print("What are you interested in? \n 1 - Trip duration \n 2 - Direction of trips \n 3 - Number of trips  \n 4 - Go back")
    userInteraction = input("Choose a number <1 - 4> \n")
    userInteraction = int(userInteraction)

    #Chose a prediction
    if userInteraction in [1,2,3]:
        predict(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,userInteraction,defaultData)
    #Go back
    elif userInteraction == 4:
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)
    #Invalid input
    else:
        print("Invalid input")
        menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)


#Load the last builded datframes
def useExisting():

    dfWeather = io.readSavedWeather()
    dfTrips = io.readSavedTrips()
    dfStations = io.readSavedStations()
    dfBikesPerStationIndex = io.readSavedBikesPerStation()
    dfTripsPerDay = io.readSavedTripsPerDay()

    return dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay


#Rebuild all dataframes based on the default or a custom file
def rebuild(datapath = None):

    print("This step will take a few minutes.")

    #Def reload all data 
    print("Get raw data    -- 0%")
    if datapath != None:
        rawData = io.read_file(datapath)
    else:
        rawData = io.read_file()
    dfWeather = io.getWeatherData()

    #Preprocess Trips
    print("create trips     -- 10% ")
    dfRawData = io.preprocessData(rawData)  
    dfTripsRaw = io.createTrips(dfRawData)
    dfTrips = io.drop_outliers(dfTripsRaw)

    #Add postalcode infos
    print("Assign postalCode   -- 60%")
    dfTrips = postalCodes.assignPostalCode(dfTrips)
    dfTrips = postalCodes.filterForPostalCodes(dfTrips)
    

    #Create station data
    print("Get station data   -- 80%")
    stationData = io.preprocessStationData(rawData)
    dfBikesPerStationIndex = io.createBikeNumberPerStationIndex(stationData)
    dfStations = io.createStations(dfRawData)

    #For additional predictions
    print("Trips per day   -- 90%")
    dfTripsPerDay = io.createTripsPerDay(dfTrips,dfWeather)


    #Save dataframes for system reuse
    print("Save Dataframes")
    io.save_WeatherForReues(dfWeather)
    io.save_tripDataForReues(dfTrips)
    io.save_StationDataForReues(dfStations)
    io.save_dfBikesPerStationIndexForReues(dfBikesPerStationIndex)
    io.save_tripsPerDayForReues(dfTripsPerDay)

    #Save dataframes as output
    io.save_Weather(dfWeather)
    io.save_tripData(dfTrips)
    io.save_StationData(dfStations)
    io.save_dfBikesPerStationIndexs(dfBikesPerStationIndex)
    io.save_tripsPerDay(dfTripsPerDay)

    return dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay


#Start a specific visualization topic, based on the users choice
def visualize(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,inputType,defaultData):

    #Visualizations about trip duration
    if inputType == 1:
        visualization.visualizeMeanTripLength(dfTrips).show()
        visualization.visualizeStdTripLength(dfTrips).show()
        visualization.visualizeTripLengthBoxplots(dfTrips)
        visualization.visualizeDistributionOfTripsPerMonth(dfTrips).show()
    
    #Visualization about the number of trips
    elif inputType == 2:
        visualization.visualizeNumberOfTrips(dfTrips).show()

    #Visualizations about bikes per station
    #User has to define a timestamp
    elif inputType == 4:
        year = 2019
        month = int(input("Chose a month <1-12> \n"))
        day = int(input("Chose a day \n"))
        hour = int(input("Chose a hour <0-23> \n"))
        minute = int(input("Chose a minute <0-59> \n"))
        secound = 0

        pointInTime = datetime.datetime(year,month,day,hour,minute,secound)
        visualization.visualizeNumberOfBikesPerStationMap(pointInTime, dfStations, dfBikesPerStationIndex)
        visualization.visualizeNumberOfBikesPerStationBarplot(pointInTime, dfStations, dfBikesPerStationIndex)

    #Visualization about weather data
    elif inputType == 5:
        visualization.visualizeWeatherData(dfWeather).show()
        visualization.visualizeNumberOfTripsWithTemperatureAndPrecipitation(dfWeather,dfTrips)

    #Visualization about trip starts/end in postalcode area
    elif inputType == 3:
        while True:

            print("Do you want to see a interactive map, showing the start/end postal code region of trips per month?")
            userInput = input("<y,n> \n")

            if userInput.upper() == 'Y':
                print("Do you want to see 1 - start or 2 - end postal code area?")
                userInput1 = input("<1,2> \n")

                print("Choose a month")
                userInput2 = input("<1-12> \n")

                if userInput1 == '1':
                    postalCodes.createTripsPerPostalCodeMap(dfTrips,int(userInput2),True)
                else:
                    postalCodes.createTripsPerPostalCodeMap(dfTrips,int(userInput2),False)
                
                

            else:
                break
    
    #Visualization about connection between trips and events in marburg
    elif inputType == 6:
        visualization.visualizeEventHeatmap(dfTrips,dfStations,'2019-11-29 12:00',startOrend='end',timeframe=5,max_val=30)
        


    menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)



# Starts a specific prediction based on users choice
def predict(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,type,defaultData):

    #Trip duration
    if type == 1:
        print("Do you want to 1 - Train or 2 - Predict?")
        userInput = int(input("<1,2> \n"))

        #Train the model
        if userInput == 1:
            print('This can take a few minutes')    
            prediction.retrainModel_DurationOfTrips(dfTrips,dfWeather,False)
        
        #Load a model and predict
        elif userInput == 2:
            print('This can take a few minutes')
            model, sscaler, sscalerY = prediction.loadModel_DurationOfTrips()
            prediction.predict_DurationOfTrips(dfTrips, dfWeather, model, sscaler,sscalerY)

    #Trip direction
    elif type == 2:
        print("Do you want to 1 - Train or 2 - Predict?")
        userInput = int(input("<1,2> \n"))

        #Train the model
        if userInput == 1:
            prediction.trainKNNRegression(dfTrips,dfWeather)

        #Predict
        elif userInput == 2:
            prediction.predictTripDirection(dfTrips,dfWeather)
            
    #Number of trips per day
    elif type == 3:
        print("Do you want to 1 - Train or 2 - Predict?")
        userInput = int(input("<1,2> \n"))

        #Train the model, with or without hyperparamteroptimization
        if userInput == 1:
            print("Do you want to use Hyperparameteroptimization?")
            userInput2 = input("<y,n> \n")
                
            if userInput2.upper() == 'Y':
                    prediction.retrainModel_NumberOfTrips(dfTripsPerDay,True)
            else:
                prediction.retrainModel_NumberOfTrips(dfTripsPerDay,False)
        
        #Predict
        elif userInput == 2:
            model, sscaler, sscalerY = prediction.loadModel_NumberOfTrips()
            prediction.predict_NumberOfTrips(dfTripsPerDay, model, sscaler,sscalerY)
        
    menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,defaultData)




if __name__ == '__main__':
    main()