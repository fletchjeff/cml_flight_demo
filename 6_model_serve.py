import pandas as pd
import cdsw

#args = {"feature" : "US,2040,DCA,BOS,1630,81,399,1,16"}

from joblib import dump, load

ct = load('models/ct.joblib')
pipe = load('models/pipe.joblib')

@cdsw.model_metrics
def predict_cancelled(args):
  inputs = args['feature'].split(",")
  inputs[1] = int(inputs[1])
  inputs[4] = int(inputs[4])
  inputs[5] = int(inputs[5])
  inputs[6] = int(inputs[6])
  inputs[7] = int(inputs[7])
  inputs[8] = int(inputs[8])

  input_cols = ['OP_CARRIER','OP_CARRIER_FL_NUM','ORIGIN','DEST','CRS_DEP_TIME','CRS_ELAPSED_TIME','DISTANCE','WEEK','HOUR']
  input_df = pd.DataFrame([inputs],columns=input_cols )

  input_transformed = ct.transform(input_df)
  
  prediction = pipe.predict(input_transformed)
  cdsw.track_metric('input_data', args)
  cdsw.track_metric('prediction', int(prediction[0]))
  
  return {
    "prediction" : int(prediction[0])
  }



