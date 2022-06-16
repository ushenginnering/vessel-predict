# {"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"name":"app.py","provenance":[],"authorship_tag":"ABX9TyMCTrskvOL8PotWIwe+w7ID"},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"code","execution_count":6,"metadata":{"id":"ONOY5LvWd4s7","executionInfo":{"status":"ok","timestamp":1653739128995,"user_tz":-60,"elapsed":1153,"user":{"displayName":"Ush Engineering","userId":"12457121566292976244"}}},"outputs":[],"source":["\n","def Hello_World():\n","  try:\n","    print(\"Hello World\")\n","\n","  except:\n","    pass"]},{"cell_type":"code","source":["print(f\"Tell them {Hello_World()}, thank you\")"],"metadata":{"colab":{"base_uri":"https://localhost:8080/"},"id":"Z6oT5JzyiJmF","executionInfo":{"status":"ok","timestamp":1653739301085,"user_tz":-60,"elapsed":538,"user":{"displayName":"Ush Engineering","userId":"12457121566292976244"}},"outputId":"005fd065-4cf7-4f8f-ec2a-39fd6aec83a6"},"execution_count":8,"outputs":[{"output_type":"stream","name":"stdout","text":["Hello World\n","Tell them None, thank you\n"]}]}]}

import numpy as np

from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

from flask import Flask, request, render_template
import pickle
import joblib

app = Flask(__name__)
model = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model.pkl")
# model = pickle.load(open('weight_pred_model.pkl', 'rb'))

DAYS = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/days.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getprediction',methods=['POST'])
def getprediction():    

#     input = [float(x) for x in request.form.values()]
#     final_input = [np.array(input)]
#     prediction = model.predict(final_input)

#     return render_template('index.html', output='Predicted Weight in KGs :{}'.format(prediction))

# we use this instead of commented one above
  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

# features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['Packed', 'Tonnage', 'Receiver',	'Type',	'Berth'] 
  predict = {}

  print("Please type numeric values as prompted.")
  for feature in features:
    valid = True
    while valid: 
      val = input(feature + ": ") + ".0"
      if not val.isdigit(): valid = False

    predict[feature] = [float(val) for val in request.form.values()]

  predictions = model.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS[class_id], 100 * probability))

  return render_template('index.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))
   

if __name__ == "__main__":
    app.run(debug=True)