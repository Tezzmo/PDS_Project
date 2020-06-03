import click
import datetime
from . import io
from . import model
from . import operation
from . import postalCodes
from . import visualization
from . import prediction
from IPython.display import display, Image
from IPython.core.display import HTML

@click.command()
@click.option('--start',default=False, help="Train the model.")

def main(start):

    print("Welcome")
    print("Your options: \n1 - Rebuild the model \n2 - Rebuild the model on new data \n3 - Use saved model")
    userInteraction = input("Press 1, 2 or 3") 

    if userInteraction == '1':
        print('Start to rebuild the model, this will take several minutes')
        dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild()
    elif userInteraction == '2':
        print('Input the filename in data/input/')
        fileInput = input('Exp.: data.csv')
        dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay = rebuild()
    
    elif userInteraction == '3':
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

    print("What are you interested in? \n 1 - Tripduration \n 2 - Direction of trips \n 3 - Number of trips  \n 4 - Go back")
    userInteraction = input("Choose a number <1 - 4>")
    userInteraction = int(userInteraction)

    if userInteraction in [1,2,3]:
        predict(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,userInteraction)
    elif userInteraction == 4:
        mainMenue(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)
    else:
        print("Invalid input")
        menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)



def useExisting():

    dfWeather = io.readSavedWeather()
    dfTrips = io.readSavedTrips()
    dfStations = io.readSavedStations()
    dfBikesPerStationIndex = io.readSavedBikesPerStation()
    dfTripsPerDay = io.readSavedTripsPerDay()

    return dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay


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


    #Save data
    #Pfad anpassen !!!  --> Muss in input speichern
    print("Save Dataframes")
    io.save_WeatherForReues(dfWeather)
    io.save_tripDataForReues(dfTrips)
    io.save_StationDataForReues(dfStations)
    io.save_dfBikesPerStationIndexForReues(dfBikesPerStationIndex)
    io.save_tripsPerDayForReues(dfTripsPerDay)

    return dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay


def visualize(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,type):

    if type == 1:
        visualization.visualizeMeanTripLength(dfTrips).show()
        visualization.visualizeStdTripLength(dfTrips).show()
        visualization.visualizeTripLengthBoxplots(dfTrips)
    
    elif type == 2:
        visualization.visualizeNumberOfTrips(dfTrips).show()
        visualization.visualizeDistributionOfTripsPerMonth(dfTrips).show()

    elif type == 4:
        year = 2019
        month = int(input("Chose a month <1-12>"))
        day = int(input("Chose a day"))
        hour = int(input("Chose a hour <0-23>"))
        minute = int(input("Chose a minute <0-59>"))
        secound = 0

        pointInTime = datetime.datetime(year,month,day,hour,minute,secound)

        visualization.visualizeNumberOfBikesPerStationMap(pointInTime, dfStations, dfBikesPerStationIndex)
        
        visualization.visualizeNumberOfBikesPerStationBarplot(pointInTime, dfStations, dfBikesPerStationIndex)

    elif type == 5:
        visualization.visualizeWeatherData(dfWeather).show()

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
                    postalCodes.createTripsPerPostalCodeMap(dfTrips,int(userInput2),True)
                else:
                    postalCodes.createTripsPerPostalCodeMap(dfTrips,int(userInput2),False)
                
                

            else:
                break

    menueVisualization(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)




def predict(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay,type):

    if type == 1:
        pass

    elif type == 2:
        pass

    elif type == 3:
        print("Do you want to 1 - Train or 2 - Predict?")
        userInput = int(input("<1,2>"))

        if userInput == 1:
            print("Do you want to use Hyperparameteroptimization?")
            userInput2 = input("<y,n>")
                
            if userInput2.upper() == 'Y':
                    prediction.retrainModel_NumberOfTrips(dfTripsPerDay,True)
            else:
                prediction.retrainModel_NumberOfTrips(dfTripsPerDay,False)
        elif userInput == 2:
            model, sscaler, sscalerY = prediction.loadModel_NumberOfTrips()
            prediction.predict_NumberOfTrips(dfTripsPerDay, model, sscaler,sscalerY)
        
        menuePrediction(dfWeather,dfTrips,dfStations,dfBikesPerStationIndex,dfTripsPerDay)




if __name__ == '__main__':
    main()