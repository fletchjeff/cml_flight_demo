import cdsw, time, os
import pandas as pd
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap

model_id = "33"

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

# Read in the model metrics dict.
model_metrics = cdsw.read_metrics(model_crn=Model_CRN,model_deployment_crn=Deployment_CRN)

# This is a handy way to unravel the dict into a big pandas dataframe.
metrics_df = pd.io.json.json_normalize(model_metrics["metrics"])

latest_aggregate_metric = metrics_df.dropna(subset=["metrics.accuracy"]).sort_values('startTimeStampMs')[-1:]["metrics.accuracy"]


if latest_aggregate_metric.to_list()[0] < 0.6:
  print("model is below threshold, retraining")
  cml.start_job(68,{})
  #TODO reploy new model
else:
  print("model does not need to be retrained")
  
