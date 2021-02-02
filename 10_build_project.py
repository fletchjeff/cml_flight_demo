# Run this file to auto deploy the model, run a job, and deploy the application

# Install the requirements
!pip3 install -U pip
!pip3 install -r requirements.txt --progress-bar off

import subprocess
import datetime
import xml.etree.ElementTree as ET
import requests
import json
import time
import os
from IPython.display import Javascript, HTML
from cmlbootstrap import CMLBootstrap


# Instantiate API Wrapper
HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split("/")[6]
API_KEY = os.getenv("CDSW_API_KEY")
PROJECT_NAME = os.getenv("CDSW_PROJECT")

cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

# Set the STORAGE environment variable
try : 
  storage=os.environ["STORAGE"]
except:
  if os.path.exists("/etc/hadoop/conf/hive-site.xml"):
    tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
    root = tree.getroot()
    for prop in root.findall('property'):
      if prop.find('name').text == "hive.metastore.warehouse.dir":
        storage = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  else:
    storage = "/user/" + os.getenv("HADOOP_USER_NAME")
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage

# Unzip data
!mkdir data
!cp all_flight_data.tgz data
!cd data && tar xjvf all_flight_data.tgz

!hdfs dfs -mkdir -p $STORAGE/datalake
!hdfs dfs -mkdir -p $STORAGE/datalake/data
!hdfs dfs -mkdir -p $STORAGE/datalake/data/flight_data
!hdfs dfs -mkdir -p $STORAGE/datalake/data/flight_data/set_1
!hdfs dfs -mkdir -p $STORAGE/datalake/data/flight_data/set_2

!curl https://cdp-demo-data.s3-us-west-2.amazonaws.com/all_flight_data.zip | zcat | hadoop fs -put - $STORAGE/datalake/data/flight_data/set_1/flight_data_1.csv
!for i in $(seq 2009 2018); do curl https://cdp-demo-data.s3-us-west-2.amazonaws.com/$i.csv | hadoop fs -put - $STORAGE/datalake/data/flight_data/set_2/$i.csv; done


# Get User Details
user_details = cml.get_user({})
user_obj = {"id": user_details["id"], "username": USERNAME,
            "name": user_details["name"],
            "type": user_details["type"],
            "html_url": user_details["html_url"],
            "url": user_details["url"]
            }

# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]

# Create Train Job
run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

create_jobs_params = {"name": "Train Model " + run_time_suffix,
                      "type": "manual",
                      "script": "5_model_train.py",
                      "timezone": "America/Los_Angeles",
                      "environment": {},
                      "kernel": "python3",
                      "cpu": 4,
                      "memory": 8,
                      "nvidia_gpu": 0,
                      "include_logs": True,
                      "notifications": [
                          {"user_id": user_obj["id"],
                           "user":  user_obj,
                           "success": False, "failure": False, "timeout": False, "stopped": False
                           }
                      ],
                      "recipients": {},
                      "attachments": [],
                      "include_logs": True,
                      "report_attachments": [],
                      "success_recipients": [],
                      "failure_recipients": [],
                      "timeout_recipients": [],
                      "stopped_recipients": []
                      }

new_job = cml.create_job(create_jobs_params)
new_job_id = new_job["id"]
print("Created new job with jobid", new_job_id)

# Start Train Job
job_env_params = {}
start_job_params = {"environment": job_env_params}
job_id = new_job_id
job_status = cml.start_job(job_id, start_job_params)
print("Job started")

# Get Default Engine Details
default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]

# Create Model
example_model_input = {"feature": "US,2040,DCA,BOS,1630,81,399,16,16"} # to-do

create_model_params = {
    "projectId": project_id,
    "name": "Flight Model " + run_time_suffix,
    "description": "Explain a given model prediction",
    "visibility": "private",
    "enableAuth": False,
    "targetFilePath": "6_model_serve.py",
    "targetFunctionName": "predict_cancelled",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": [
        {
            "request": example_model_input,
            "response": {}
        }],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}

new_model_details = cml.create_model(create_model_params)
access_key = new_model_details["accessKey"]  # todo check for bad response
model_id = new_model_details["id"]

print("New model created with access key", access_key)

# Disable model_authentication
cml.set_model_auth({"id": model_id, "enableAuth": False})

# Wait for the model to deploy.
is_deployed = False
while is_deployed == False:
    model = cml.get_model({"id": str(
        new_model_details["id"]), "latestModelDeployment": True, "latestModelBuild": True})
    if model["latestModelDeployment"]["status"] == 'deployed':
        print("Model is deployed")
        break
    else:
        print("Deploying Model.....")
        time.sleep(10)

# Create Application
create_application_params = {
    "name": "Flight Prediction App",
    "subdomain": run_time_suffix[:],
    "description": "Web App to Display Flight Predictions",
    "type": "manual",
    "script": "7_application.py", 
    "environment": {"SHTM_ACCESS_KEY": access_key},
    "kernel": "python3", 
    "cpu": 1, 
    "memory": 2,
    "nvidia_gpu": 0
}

new_application_details = cml.create_application(create_application_params)
application_url = new_application_details["url"]
application_id = new_application_details["id"]

# print("Application may need a few minutes to finish deploying. Open link below in about a minute ..")
print("Application created, deploying at ", application_url)

# Wait for the application to deploy.
is_deployed = False
while is_deployed == False:
    # Wait for the application to deploy.
    app = cml.get_application(str(application_id), {})
    if app["status"] == 'running':
        print("Application is deployed")
        break
    else:
        print("Deploying Application.....")
        time.sleep(10)

HTML("<a href='{}'>Open Application UI</a>".format(application_url))

# This will run the model operations section that makes calls to the model to track
# mertics and track metric aggregations
exec(open("8_model_simulation.py").read())