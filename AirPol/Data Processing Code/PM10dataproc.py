# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:35:21 2020

@author: hanan
"""

# Import needed libraries
#import customfunctions as cf # a Python file with functions I wrote for repetitive tasks
import os
import pandas as pd
from numpy import array
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#import plotly.express as px

# Read in the data (multiple CSV files, 1 file per year of data from 1980 to 2019)
pm10datadir = r'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM10_mass/Raw Data'
pm10df = pd.DataFrame()
for f in os.listdir(pm10datadir):
    if f.endswith(".csv"):
        pm10df = pm10df.append(pd.read_csv(os.path.join(pm10datadir, f), parse_dates = ['Date Local'], infer_datetime_format = True, squeeze = True, usecols = ['Date Local', 'Arithmetic Mean'], encoding = 'utf-8-sig', low_memory = False)[['Date Local', 'Arithmetic Mean']], ignore_index = True)

# Get info about the data
#print("Info about the PM10 data(raw): \n%s\n" % pm10df.info())
#print("The first 5 rows of the PM10 data(raw):\n%s\n" % pm10df.head())
#print("The last 5 rows of the PM10 data(raw):\n%s\n" % pm10df.tail())
        
# Perform data cleaning on the Pandas dataframes as needed
pm10df.sort_values(by = ['Date Local'], ascending = True, inplace = True, kind = 'mergesort', ignore_index = True) # Sort the rows by date in ascending order
#print("Info about the PM10 data(sorted): \n%s\n" % pm10df.info())
#print("The first 5 rows of the PM10 data(sorted): \n%s\n" % pm10df.head())
#print("The last 5 rows of the PM10 data(sorted): \n%s" % pm10df.tail())

# Creating a new dataframe with the average daily concentration of PM10 and the corresponding date
# Calculate the mean for each day
pm10means = []
for u in pm10df['Date Local'].unique():
    pm10means.append(pm10df[pm10df['Date Local'] == u].mean())

# Setting up a dictionary containing the new data
pm10rd = {'Date': pm10df['Date Local'].unique(), 'Average_PM10_Concentration': pm10means}    

# Converting the dictionary into a pandas Dataframe
pm10_finalDF = pd.DataFrame(pm10rd, columns = ['Date', 'Average_PM10_Concentration'])
for d in pm10_finalDF['Date']: # Setting the date format to YYYY-MM-DD
    pm10_finalDF.loc[pm10_finalDF.Date == d, 'Date'] = d.strftime('%Y-%m-%d')
pm10_finalDF['Average_PM10_Concentration'] = pm10_finalDF['Average_PM10_Concentration'].astype(str) # Converting the values of this column into strings (for use with regex)   
pm10_finalDF['Average_PM10_Concentration'] = pm10_finalDF['Average_PM10_Concentration'].str.extract('(\d*\.\d*)').astype('float64') # Extracting only the number from the string and converting it to a float  
print("Info about the PM10 data(clean): \n%s\n" % pm10_finalDF.info())
print("The first 5 rows of the PM10 data(clean):\n%s\n" % pm10_finalDF.head())
print("The last 5 rows of the PM10 data(clean):\n%s" % pm10_finalDF.tail())

# Checking for the folder to store the cleaned data in & creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM10_mass/Clean Data'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM10_mass/Clean Data')

# Saving the cleaned data as a CSV file
cleaned_pm10csv = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM10_mass/Clean Data/cleaned_PM10Data.csv'
pm10_finalDF.to_csv(cleaned_pm10csv, date_format = '%Y-%m-%d')

# Checking for the folder that plotly figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures')

'''    
# Plotting the data used to train the model (the daily average concentration of PM10 in µg/m^3 - micrograms per cubic meter)
pm10_fig = px.scatter(pm10_finalDF, x = 'Date', y = 'Average_PM10_Concentration', width = 3000, height = 2500)
pm10_fig.add_trace(go.Scatter(
    x = pm10_finalDF['Date'],
    y = pm10_finalDF['Average_PM10_Concentration'],
    name = 'PM10',
    line_color = 'red',
    opacity = 0.8  
))
pm10_fig.update_layout(
    xaxis_range = ['1980-01-01', '2019-12-31'], 
    title_text = 'US Daily Avg. PM10 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (µg/m^3)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
pm10_fig.update_xaxes(automargin = True)
pm10_fig.update_yaxes(automargin = True)
pm10_fig.write_image('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures/avg_pm10.png')
'''

# Split the data into the train/test sets based on the date
pm10_masktrain = (pm10_finalDF['Date'] < '2019-01-01')
pm10_masktest = (pm10_finalDF['Date'] >= '2019-01-01')
pm10_train, pm10_test = pm10_finalDF.loc[pm10_masktrain], pm10_finalDF.loc[pm10_masktest]

# Set up the Keras TimeSeriesGenerator for converting the data into a form recognizable by the model
ser_train = array(pm10_train['Average_PM10_Concentration'].values)
ser_test = array(pm10_test['Average_PM10_Concentration'].values)
n_feat = 1
ser_train = ser_train.reshape((len(ser_train), n_feat))
n_in = 2
train_gen = TimeseriesGenerator(ser_train, ser_train, length = n_in, sampling_rate = 1, batch_size = 10)
test_gen = TimeseriesGenerator(ser_test, ser_test, length = n_in, sampling_rate = 1, batch_size = 1)
print('Number of training samples: %d\n' % len(train_gen))
print('Number of testing samples: %d\n' % len(test_gen))

# Defining an alternate optimizer (in case the ADAM optimizer doesn't work well for this application)
opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)

# Defining the model's structure
pm10_mod = Sequential([
    LSTM(50, activation = 'relu', input_shape = (n_in, n_feat), return_sequences = True),
    Dropout(0.2),
    LSTM(50, return_sequences = True),
    Dropout(0.2),
    LSTM(50, return_sequences = True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compiling & fitting the model
pm10_mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = pm10_mod.fit_generator(
    train_gen, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Show a summary of the model
print("Summary of the model: %s\n" % pm10_mod.summary())

# Creating the folder to save models in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models')
    
# Save the model as an HDF5 object (as an .h5 file)
path = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models/pm10_model.h5'
pm10_mod.save(path, overwrite = True)

# Make a test prediction
x_in = array(pm10_test['Average_PM10_Concentration'].head(n_in)).reshape((1, n_in, n_feat))
pm10_pred = pm10_mod.predict(x_in, verbose = 0)
print('\nPredicted daily avg. PM10 concentration: %.3f µg/m^3\n' % pm10_pred[0][0])
print(pm10_finalDF[pm10_finalDF['Date'] == '2019-01-03'])

# Creating the folder to save matplotlib figures in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures')

# Plotting the model's performance metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('PM10 Data LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
plt.legend()
plt.savefig('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures/pm10_modelmetrics.png')