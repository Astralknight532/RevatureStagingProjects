from flask import Flask, render_template, request
import tensorflow as tf
import keras
from keras.models import load_model
import pandas as pd
import numpy as np
from datetime import timedelta

app = Flask(__name__)

'''
# Loading the models
# NO2 model
no2_model = load_model('C:/Users/hanan/Desktop/StagingProjects/AirPol/SavedModels/no2_model.h5') 
# SO2 model
so2_model = load_model('C:/Users/hanan/Desktop/StagingProjects/AirPol/SavedModels/so2_model.h5') 
# O3 model
o3_model = load_model('C:/Users/hanan/Desktop/StagingProjects/AirPol/SavedModels/o3_model.h5')
# CO model
co_model = load_model('C:/Users/hanan/Desktop/StagingProjects/AirPol/SavedModels/co_model.h5')
'''

@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict_result():
    date = request.form["dateentry"] # Get the date entered by the user
    pol = request.form["polselect"] # Get the pollutant chosen by the user 
    avgconc = 0 # Create a variable to store the predicted avg. concentration for a pollutant
    date = pd.to_datetime(date, format = '%Y-%m-%d')
    pred_input = np.array([date - timedelta(days = 2), date - timedelta(days = 1)])
    pred_input = np.array(pd.DataFrame(pred_input)).reshape((1, 2, 1))

    '''
    # Select the appropriate model based on the user's chosen pollutant
    if pol == 'NO2':
        print('NO2 Model')
        avgconc = no2_model.predict(pred_input)[0][0]
    elif pol == 'SO2':
        print('SO2 Model')
        avgconc = so2_model.predict(pred_input)[0][0]
    elif pol == 'O3':
        print('O3 Model')
        avgconc = o3_model.predict(pred_input)[0][0]
    elif pol == 'CO':
        print('CO Model')
        avgconc = co_model.predict(pred_input)[0][0]
    '''
    
    avgconc_print = str(avgconc)
    if pol == 'NO2' or pol == 'SO2':
        avgconc_print += ' parts per billion'
    elif pol == 'O3' or pol == 'CO':
        avgconc_print += ' parts per million'
    
    return render_template("results.html", chosendate = str(date.strftime('%Y-%m-%d')), pollutant = pol, avgconc = avgconc_print)

app.run(host = '127.0.0.1', port = '5000', debug = False, threaded = False)