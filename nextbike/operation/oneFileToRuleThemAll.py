import nextbike
import datetime

def start():

    print("Welcome")
    print("Your options: \n1 - Rebuild the model \n2 - Use saved model")
    userInteraction = input("Press 1 or 2") 

    if userInteraction == '1':
        print('Start to rebuild the model, this will take several minutes')
        dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild()
    else:
        print('Start to load all data')
        dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = useExisting()

    mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)



def mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay):

    print("Your options: \n1 - Visualize \n2 - Predict \n3 - End")
    userInteraction = input("Press 1 or 2") 

    if userInteraction == '1':
        menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)

    elif userInteraction == '2':
        menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)

    elif userInteraction == '3':
        pass

    else:
        print("Invalied Input")
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)




def menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay):

    print("What are you interested in? \n 1 - Tripduration \n 2 - Number of Trips \n 3 - Start/End point of Trips \n 4 - Bikes per Station \n 5 - Weather data \n 6 - Go back")
    userInteraction = input("Choose a number <1 - 6>")
    userInteraction = int(userInteraction)

    if userInteraction in [1,2,3,4,5,6]:
        visualize(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,userInteraction)
    elif userInteraction == 6:
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)
    else:
        print("Invalid input")
        menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)
        



def menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay):

    print("What are you interested in? \n 1 - Tripduration \n 2 - Number of Trips \n 3 - Number of Trips  \n 4 - Go back")
    userInteraction = input("Choose a number <1 - 4>")
    userInteraction = int(userInteraction)

    if userInteraction in [1,2,3]:
        prediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,userInteraction)
    elif userInteraction == 4:
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)
    else:
        print("Invalid input")
        menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)



def useExisting():

    dfWeather = nextbike.io.readSavedWeather()
    dfTrips = nextbike.io.readSavedTrips()
    dfStations = nextbike.io.readSavedStations()
    dfBikesPerStationIndex = nextbike.io.readSavedBikesPerStation()
    dfTripsPerDay = nextbike.io.readSavedTripsPerDay()

    return dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay


def rebuild():

    print("This step will take a few minutes.")

    #Def reload all data 
    print("Get raw data    -- 0%")
    rawData = nextbike.io.read_file()
    dfWeather = nextbike.io.getWeatherData()

    #Preprocess Trips
    print("create trips     -- 10% ")
    dfRawData = nextbike.io.preprocessData(rawData)  
    dfTripsRaw = nextbike.io.createTrips(dfRawData)
    dfTrips = nextbike.io.drop_outliers(dfTripsRaw)

    #Add postalcode infos
    print("Assign postalCode   -- 60%")
    dfTrips = nextbike.postalCodes.assignPostalCode(dfTrips)
    dfTrips = nextbike.postalCodes.filterForPostalCodes(dfTrips)
    

    #Create station data
    print("Get station data   -- 80%")
    stationData = nextbike.io.preprocessStationData(rawData)
    dfBikesPerStationIndex = nextbike.io.createBikeNumberPerStationIndex(stationData)
    dfStations = nextbike.io.createStations(dfRawData)

    #For additional predictions
    print("Trips per day   -- 90%")
    dfTripsPerDay = nextbike.io.createTripsPerDay(dfTrips,dfWeather)


    #Save data
    #Pfad anpassen !!!  --> Muss in input speichern
    print("Save Dataframes")
    nextbike.io.save_WeatherForReues(dfWeather)
    nextbike.io.save_tripDataForReues(dfTrips)
    nextbike.io.save_StationDataForReues(dfStations)
    nextbike.io.save_dfBikesPerStationIndexForReues(dfBikesPerStationIndex)
    nextbike.io.save_tripsPerDayForReues(dfTripsPerDay)

    return dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay


def visualize(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,type):

    if type == 1:
        nextbike.visualization.visualizeMeanTripLength(dfTrips).show()
        nextbike.visualization.visualizeStdTripLength(dfTrips).show()
        nextbike.visualization.visualizeTripLengthBoxplots(dfTrips)
    
    elif type == 2:
        nextbike.visualization.visualizeNumberOfTrips(dfTrips).show()
        nextbike.visualization.visualizeDistributionOfTripsPerMonth(dfTrips).show()

    elif type == 4:
        year = 2019
        month = int(input("Chose a month <1-12>"))
        day = int(input("Chose a day"))
        hour = int(input("Chose a hour <0-23>"))
        minute = int(input("Chose a minute <0-59>"))
        secound = 0

        pointInTime = datetime.datetime(year,month,day,hour,minute,secound)

        nextbike.visualization.visualizeNumberOfBikesPerStationMap(pointInTime, dfStations, dfBikesPerStationIndex).show()
        nextbike.visualization.visualizeNumberOfBikesPerStationBarplot(pointInTime, dfStations, dfBikesPerStationIndex).show()

    elif type == 5:
        nextbike.visualization.visualizeWeatherData(dfWeather).show()

    elif type == 3:
        while True:

            print("Do you want to see a interactive map, showing the start/end postal code region of trips per month?")
            userInput = input("<y,n>")

            if userInput.upper() == 'Y':
                print("Do you want to see 1 - start or 2 - end postal code area?")
                userInput1 = input("<1,2>")

                print("Choose a month")
                userInput2 = input("<1-12>")

                if userInput1 == '1':
                    map = nextbike.postalCodes.createTripsPerPostalCodeMap(dfTrips,int(userInput2),True)
                else:
                    map = nextbike.postalCodes.createTripsPerPostalCodeMap(dfTrips,int(userInput2),False)
                
                display(map)

            else:
                break

    menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)




def prediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,type):

    if type == 1:
        pass

    elif type == 2:
        pass

    elif type == 3:
        print("Do you want to 1 - retrain the model or 2 - load a existing?")
        userInput = int(input("<1,2>"))

        if userInput == 1:
            print("Do you want to use Hyperparameteroptimization?")
            userInput2 = input("<y,n>")
                
            if userInput2.upper() == 'Y':
                    nextbike.prediction.retrainModel_NumberOfTrips(dfTripsPerDay,True)
            else:
                nextbike.prediction.retrainModel_NumberOfTrips(dfTripsPerDay,False)

        else:
            pass


