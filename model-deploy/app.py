import os

!pip install -q colabcode #REMEMBER
!pip install jinja2==2.11.3
!pip install flask
#!pip install flask-ngrok

from colabcode import ColabCode

#no Auth please

import numpy as np

from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

#from flask_ngrok import run_with_ngrok
from flask import Flask, request, render_template
import pickle
import joblib

app = Flask(__name__)

#run_with_ngrok(app)
model = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_bi.pkl")
# model = pickle.load(open('weight_pred_model.pkl', 'rb'))

model1 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_gc.pkl")

model2 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_gp.pkl")

model3 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_mt.pkl")

model4 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_mtgc.pkl")

model5 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_st.pkl")

model6 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_wt.pkl")

model7 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/model_zc.pkl")

DAYS = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/daysbi.pkl")

DAYS1 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/daysgc.pkl")

DAYS2 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/daysgp.pkl")

DAYS3 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/daysmt.pkl")

DAYS4 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/daysmtgc.pkl")

DAYS5 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/daysst.pkl")

DAYS6 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/dayswt.pkl")

DAYS7 = joblib.load("/content/drive/MyDrive/Colab Notebooks/Vessel Project/ml-web-app/model-deploy/dayszc.pkl")

@app.route('/')
def home():
    return render_template('/startup2-1.0.0/iindex.html')
    # return redirect(url_for('/startup2-1.0.0/iindex.html'))

@app.route('/quote')
def quote():
    return render_template('/startup2-1.0.0/quote.html')

@app.route('/contact')
def contact():
    return render_template('/startup2-1.0.0/contact.html')

@app.route('/getprediction',methods=['POST'])
def getprediction():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  # print("Please type numeric values as prompted.")

  keys = request.form.keys()
  # keys = sorted(keys)

  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)
      # print(key, request.form.get(key))

  predictions = model.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      # print('Prediction is "{} Days" ({:.1f}%)'.format(
      #     DAYS[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))
  # return redirect(url_for('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))

# General Cargo session
@app.route('/getprediction1',methods=['POST'])
def getprediction1():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()

  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)

  predictions = model1.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS1[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS1[class_id], 100 * probability))

# Gypsum session
@app.route('/getprediction2',methods=['POST'])
def getprediction2():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)

  predictions = model2.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS2[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS2[class_id], 100 * probability))

# Malt session
@app.route('/getprediction3',methods=['POST'])
def getprediction3():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)

  predictions = model3.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS3[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS3[class_id], 100 * probability))

# Malt & General Cargo session
@app.route('/getprediction4',methods=['POST'])
def getprediction4():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)

  predictions = model4.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS4[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS4[class_id], 100 * probability))

# Salt session
@app.route('/getprediction5',methods=['POST'])
def getprediction5():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)

  predictions = model5.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS5[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS5[class_id], 100 * probability))

# Wheat session
@app.route('/getprediction6',methods=['POST'])
def getprediction6():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)

  predictions = model6.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS6[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS6[class_id], 100 * probability))

# Zinc session
@app.route('/getprediction7',methods=['POST'])
def getprediction7():    

  def input_fn(features, batch_size=256):
      #Convert the inputs to a Dataset without labels.
      return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

      # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
      print(key, request.form[key])
      for i in request.form[key]:
        predict.setdefault(key, []).append(float(i))
        print(predict)

  predictions = model7.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
      class_id = pred_dict['class_ids'][0]
      probability = pred_dict['probabilities'][class_id]

      print('Prediction is "{} Days" ({:.1f}%)'.format(
          DAYS7[class_id], 100 * probability))

  # return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS7[class_id], 100 * probability))
  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" '.format(DAYS7[class_id]))

app.run()