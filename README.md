# Interview challenge - wattx AI / ML candidates

# 1) Occupancy prediction model

The challenge included predicting meeting room occupancy using the dataset provided in "device_activations.csv" spanning two months, from July 2022 to August 2022. This dataset comprises three core columns: 'time', 'device', and 'device_activated'.

'time': Represents the timestamp of sensor readings.
'device': Identifies the sensor in each meeting room.
'device_activated': Binary indicator (1 for occupied, 0 for unoccupied).

Solution Approach:
The solution was structured as follows:

In the solution, the time is first converted to each hour to match the data which will be used for production and as there were no values with 0 in device_activated column, the remaining time with device combination were filled in the data.

Then the data was preprocessed and split. Analteration was made to the target label 'device_activated', where '1' (room occupied) was change to the negative class ('0'), and '0' (room vacant) was reframed as the positive class ('1'). Itw was done because it is important to turn off ventilator when meeting room is not occupied to save the cost. It was done to train the model only.

Further f1 score is used as a measure, because of the imbalanced data set. as approx 20% of data was negative class and remaining 80% was positive class. Currently F1 score is .89.

After the model is trained the preiction is done for the next 24 Hr for given timestamp and csv is generated with three column time,device,activation_predicted.
'time': Represents the timestamp of sensor readings.
'device': Identifies the sensor in each meeting room.
'activation_predicted': Binary indicator (1 for occupied, 0 for unoccupied).

### Running the solution 

1) Install the requirements
   pip install -r requirements.txr

2) Run the solution

    python solution.py "2022-08-31 23:59:59" data/device_activations.csv myresult.csv

    Then you will get the file `myresult.csv` and also the trained model will be saved.
	

### Running the app locally

This app loads the model that serves predictions that predicts the occupancy of one or several 
rooms for the next 24 hours based on this pre-trained model.

Instructions:
1) Install the requirements.txt
 pip install -r requirements.txt

2)Run the app
  python app.py

3) Send a Json Post request on the app URL
   http://127.0.0.1:5000

    Example JSON request
    {
    "time": "2022-08-30 23:59:59",       #timestamp
    "device": ["device_1", "device_7"]   #devices
    }

4) Output will be a Json response with next 24 hours predictions for each device in request


### Dockerized REST API

This app loads the model that serves predictions that predicts the occupancy of one or several 
rooms for the next 24 hours based on this model trained in first part.

Running the app

1) Load the docker image
   docker load -i prediction-app.tar

2) Run the image 
   docker run -p 5000:5000 prediction-app:latest

3) Make prediction
   Send a Json Post request on the app URL
   http://localhost:5000

    Example JSON request
    {
    "time": "2022-08-30 23:59:59",       #timestamp
    "device": ["device_1", "device_7"]   #devices
    }

4) Output will be a Json response with next 24 hours predictions for each device in request