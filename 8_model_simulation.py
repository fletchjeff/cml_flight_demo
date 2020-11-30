import cdsw, time, os, random, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns
import copy
import time


## Set the model ID from deployed model

flights_data_df = pd.read_csv("data/all_flight_data.csv")

# Get the various Model CRN details
HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split("/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

latest_model = cml.get_models({"latestModelDeployment": True, "latestModelBuild": True})[0] # Note - this assumes 10_build_project.py is run before any models are built

Model_CRN = latest_model["crn"]
Deployment_CRN = latest_model["latestModelDeployment"]["crn"]

while True:
  predicted_result = []
  actual_result = []
  start_time_ms = int(round(time.time() * 1000))
  for i in range(100):
    input_data = flights_data_df.sample(n=1)[[
      'uniquecarrier',
      'flightnum',
      'origin',
      'dest',
      'crsdeptime',
      'crselapsedtime',
      'distance',
      'hour',
      'week',
      'cancelled']].to_numpy()[0]
    
    try:

      input_data[5] = int(input_data[5])
      input_data[6] = int(input_data[6])

      input_data_string = ""
      for record in input_data[:-1]:
        input_data_string = input_data_string + str(record) + ","

      input_data_string = input_data_string[:-1]
      response = cdsw.call_model(latest_model["accessKey"],{"feature" : input_data_string})

      predicted_result.append(response["response"]["prediction"]["prediction"])
      actual_result.append(input_data[-1:][0])
      cdsw.track_delayed_metrics({"actual_result":input_data[-1:][0]}, response["response"]["uuid"])
      print(str(i) + " adding " + input_data_string)
    except:
      print("invalid row")
    time.sleep(0.2)
  end_time_ms = int(round(time.time() * 1000))
  accuracy = classification_report(actual_result,predicted_result,output_dict=True)['accuracy']
  cdsw.track_aggregate_metrics({"accuracy": accuracy}, start_time_ms , end_time_ms, model_deployment_crn=Deployment_CRN)
  print("adding accuracy measure of" + str(accuracy))
