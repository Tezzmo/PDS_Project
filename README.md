# PDS_Project
 
We obtained bikesharing data for the city of Marburg in 2019. For analysis and prediction based on this data, a python project providing a command line interface is the outcome.

The command line interface provides the possibility to create, load, save and present data, visualizations and prediction models.

To start the command line interface, use the command prompt to navigate to the main folder of this project on your local computer. Next, please use the command ```pip install -e .```. After installing the package successfully, use the command ```nextbike``` to start the program.

A much more detailed explanation of the navigation in the menu of the of the program and usage of all the different possibilites is provided in the user manual, provided in this git repository, too.

To use it only with a single command line, write ```nextbike --train <filename.csv> --test <filname.csv> --predict <filename.csv>```. The data must be in the data/input directory.
Example that trains the model on old data and transforms and predicts on new data: ```nextbike --train inputData.csv --transform marburg_test.csv --predict marburg_test.csv```