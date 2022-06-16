import os
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

#from google.colab import drive
#drive.mount('/content/drive')

import joblib

model = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model.pkl")
modd = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/days.pkl")

#print(type(modd))

DAYS = modd

# Prediction point

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

  predict[feature] = [float(val)]

predictions = model.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(
        DAYS[class_id], 100 * probability))