# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:35:04 2020

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
pm25datadir = r'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM2-5_mass/Raw Data'
pm25df = pd.DataFrame()
for f in os.listdir(pm25datadir):
    if f.endswith(".csv"):
        pm25df = pm25df.append(pd.read_csv(os.path.join(pm25datadir, f), parse_dates = ['Date Local'], infer_datetime_format = True, squeeze = True, usecols = ['Date Local', 'Arithmetic Mean'], encoding = 'utf-8-sig', low_memory = False)[['Date Local', 'Arithmetic Mean']], ignore_index = True)

# Get info about the data
#print("Info about the PM2.5 data(raw): \n%s\n" % pm25df.info())
#print("The first 5 rows of the PM2.5 data(raw):\n%s\n" % pm25df.head())
#print("The last 5 rows of the PM2.5 data(raw):\n%s\n" % pm25df.tail())
        
# Perform data cleaning on the Pandas dataframes as needed
pm25df.sort_values(by = ['Date Local'], ascending = True, inplace = True, kind = 'mergesort', ignore_index = True) # Sort the rows by date in ascending order
#print("Info about the PM2.5 data(sorted): \n%s\n" % pm25df.info())
#print("The first 5 rows of the PM2.5 data(sorted): \n%s\n" % pm25df.head())
#print("The last 5 rows of the PM2.5 data(sorted): \n%s" % pm25df.tail())

# Creating a new dataframe with the average daily concentration of PM2.5 and the corresponding date
# Calculate the mean for each day
pm25means = []
for u in pm25df['Date Local'].unique():
    pm25means.append(pm25df[pm25df['Date Local'] == u].mean())

# Setting up a dictionary containing the new data
pm25rd = {'Date': pm25df['Date Local'].unique(), 'Average_PM2.5_Concentration': pm25means}    

# Converting the dictionary into a pandas Dataframe
pm25_finalDF = pd.DataFrame(pm25rd, columns = ['Date', 'Average_PM2.5_Concentration'])
for d in pm25_finalDF['Date']: # Setting the date format to YYYY-MM-DD
    pm25_finalDF.loc[pm25_finalDF.Date == d, 'Date'] = d.strftime('%Y-%m-%d')
pm25_finalDF['Average_PM2.5_Concentration'] = pm25_finalDF['Average_PM2.5_Concentration'].astype(str) # Converting the values of this column into strings (for use with regex)   
pm25_finalDF['Average_PM2.5_Concentration'] = pm25_finalDF['Average_PM2.5_Concentration'].str.extract('(\d*\.\d*)').astype('float64') # Extracting only the number from the string and converting it to a float  
print("Info about the PM2.5 data(clean): \n%s\n" % pm25_finalDF.info())
print("The first 5 rows of the PM2.5 data(clean):\n%s\n" % pm25_finalDF.head())
print("The last 5 rows of the PM2.5 data(clean):\n%s" % pm25_finalDF.tail())

# Checking for the folder to store the cleaned data in & creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM2-5_mass/Clean Data'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM2-5_mass/Clean Data')

# Saving the cleaned data as a CSV file
cleaned_pm25csv = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/PM2-5_mass/Clean Data/cleaned_PM25Data.csv'
pm25_finalDF.to_csv(cleaned_pm25csv, date_format = '%Y-%m-%d')

# Checking for the folder that plotly figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures')

'''    
# Plotting the data used to train the model (the daily average concentration of PM2.5 in µg/m^3 - micrograms per cubic meter)
pm25_fig = px.scatter(pm25_finalDF, x = 'Date', y = 'Average_PM2.5_Concentration', width = 3000, height = 2500)
pm25_fig.add_trace(go.Scatter(
    x = pm25_finalDF['Date'],
    y = pm25_finalDF['Average_PM2.5_Concentration'],
    name = 'PM2.5',
    line_color = 'red',
    opacity = 0.8  
))
pm25_fig.update_layout(
    xaxis_range = ['1980-01-01', '2019-12-31'], 
    title_text = 'US Daily Avg. PM2.5 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (µg/m^3)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
pm25_fig.update_xaxes(automargin = True)
pm25_fig.update_yaxes(automargin = True)
pm25_fig.write_image('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures/avg_pm25.png')
'''

# Split the data into the train/test sets based on the date
pm25_masktrain = (pm25_finalDF['Date'] < '2019-01-01')
pm25_masktest = (pm25_finalDF['Date'] >= '2019-01-01')
pm25_train, pm25_test = pm25_finalDF.loc[pm25_masktrain], pm25_finalDF.loc[pm25_masktest]

# Set up the Keras TimeSeriesGenerator for converting the data into a form recognizable by the model
ser_train = array(pm25_train['Average_PM2.5_Concentration'].values)
ser_test = array(pm25_test['Average_PM2.5_Concentration'].values)
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
pm25_mod = Sequential([
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
pm25_mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = pm25_mod.fit_generator(
    train_gen, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Show a summary of the model
print("Summary of the model: %s\n" % pm25_mod.summary())

# Creating the folder to save models in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models')
    
# Save the model as an HDF5 object (as an .h5 file)
path = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models/pm25_model.h5'
pm25_mod.save(path, overwrite = True)

# Make a test prediction
x_in = array(pm25_test['Average_PM2.5_Concentration'].head(n_in)).reshape((1, n_in, n_feat))
pm25_pred = pm25_mod.predict(x_in, verbose = 0)
print('\nPredicted daily avg. PM2.5 concentration: %.3f µg/m^3\n' % pm25_pred[0][0])
print(pm25_finalDF[pm25_finalDF['Date'] == '2019-01-03'])

# Creating the folder to save matplotlib figures in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures')

# Plotting the model's performance metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('PM2.5 Data LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
plt.legend()
plt.savefig('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures/pm25_modelmetrics.png')