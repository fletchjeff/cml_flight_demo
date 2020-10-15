import cdsw, time, os, random, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
import seaborn as sns
import copy
import time


## Set the model ID
# Get the model id from the model you deployed in step 5. These are unique to each 
# model on CML.

model_id = "16"

flights_data_df = pd.read_csv("data/all_flight_data.csv")

# Get the various Model CRN details
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

latest_model = cml.get_model({"id": model_id, "latestModelDeployment": True, "latestModelBuild": True})

Model_CRN = latest_model ["crn"]
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

  

#for record in json.loads(df_sample_clean.to_json(orient='records')):
#  print("Added {} records".format(percent_counter)) if (percent_counter%50 == 0) else None
#  percent_counter += 1
#  no_churn_record = copy.deepcopy(record)
#  no_churn_record.pop('customerID')
#  no_churn_record.pop('Churn')
#  # **note** this is an easy way to interact with a model in a script
#  response = cdsw.call_model(latest_model["accessKey"],no_churn_record)
#  response_labels_sample.append(
#    {
#      "uuid":response["response"]["uuid"],
#      "final_label":churn_error(record["Churn"],percent_counter/percent_max),
#      "response_label":response["response"]["prediction"]["probability"] >= 0.5,
#      "timestamp_ms":int(round(time.time() * 1000))
#    }
#  )
#
#  
#  # The "ground truth" loop adds the updated actual label value and an accuracy measure
## every 100 calls to the model.
#for index, vals in enumerate(response_labels_sample):
#  print("Update {} records".format(index)) if (index%50 == 0) else None  
#  cdsw.track_delayed_metrics({"final_label":vals['final_label']}, vals['uuid'])
#  if (index%100 == 0):
#    start_timestamp_ms = vals['timestamp_ms']
#    final_labels = []
#    response_labels = []
#  final_labels.append(vals['final_label'])
#  response_labels.append(vals['response_label'])
#  if (index%100 == 99):
#    print("Adding accuracy metrc")
#    end_timestamp_ms = vals['timestamp_ms']
#    accuracy = classification_report(final_labels,response_labels,output_dict=True)["accuracy"]
#    cdsw.track_aggregate_metrics({"accuracy": accuracy}, start_timestamp_ms , end_timestamp_ms, model_deployment_crn=Deployment_CRN)
#
