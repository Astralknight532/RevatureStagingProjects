# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:51:03 2020

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
import plotly.graph_objects as go
import plotly.express as px

# Read in the data (multiple CSV files, 1 file per year of data from 1980 to 2019)
co_datadir = r'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/CO Data/Raw Data'
co_df = pd.DataFrame()
for f in os.listdir(co_datadir):
    if f.endswith(".csv"):
        co_df = co_df.append(pd.read_csv(os.path.join(co_datadir, f), parse_dates = ['Date Local'], infer_datetime_format = True, squeeze = True, usecols = ['Date Local', 'Arithmetic Mean'], encoding = 'utf-8-sig', low_memory = False)[['Date Local', 'Arithmetic Mean']], ignore_index = True)

# Get info about the data
#print("Info about the data(raw): \n%s\n" % co_df.info())
#print("The first 5 rows of the CO data(raw):\n%s\n" % co_df.head())
#print("The last 5 rows of the CO data(raw):\n%s\n" % co_df.tail())
        
# Perform data cleaning on the Pandas dataframes as needed
co_df.sort_values(by = ['Date Local'], ascending = True, inplace = True, kind = 'mergesort', ignore_index = True) # Sort the rows by date in ascending order
#print("Info about the data(sorted): \n%s\n" % co_df.info())
#print("The first 5 rows of the CO data(sorted): \n%s\n" % co_df.head())
#print("The last 5 rows of the CO data(sorted): \n%s" % co_df.tail())

# Creating a new dataframe with the average daily concentration of CO and the corresponding date
# Calculate the mean for each day
co_means = []
for u in co_df['Date Local'].unique():
    co_means.append(co_df[co_df['Date Local'] == u].mean())

# Setting up a dictionary containing the new data
co_rd = {'Date': co_df['Date Local'].unique(), 'Average_CO_Concentration': co_means}    

# Converting the dictionary into a pandas Dataframe
co_finalDF = pd.DataFrame(co_rd, columns = ['Date', 'Average_CO_Concentration'])
for d in co_finalDF['Date']: # Setting the date format to YYYY-MM-DD
    co_finalDF.loc[co_finalDF.Date == d, 'Date'] = d.strftime('%Y-%m-%d')
co_finalDF['Average_CO_Concentration'] = co_finalDF['Average_CO_Concentration'].astype(str) # Converting the values of this column into strings (for use with regex)   
co_finalDF['Average_CO_Concentration'] = co_finalDF['Average_CO_Concentration'].str.extract('(\d*\.\d*)').astype('float64') # Extracting only the number from the string and converting it to a float  
#print("Info about the data(clean): \n%s\n" % co_finalDF.info())
#print("The first 5 rows of the CO data(clean):\n%s\n" % co_finalDF.head())
#print("The last 5 rows of the CO data(clean):\n%s" % co_finalDF.tail())

# Checking for the folder to store the cleaned data in & creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/CO Data/Clean Data'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/CO Data/Clean Data')

# Saving the cleaned data as a CSV file
cleaned_cocsv = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/CO Data/Clean Data/cleaned_COData.csv'
co_finalDF.to_csv(cleaned_cocsv, date_format = '%Y-%m-%d')

# Checking for the folder that plotly figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures')
 
# Plotting the data used to train the model (the daily average concentration of CO in PPM)
co_fig = px.scatter(co_finalDF, x = 'Date', y = 'Average_CO_Concentration', width = 3000, height = 2500)
co_fig.add_trace(go.Scatter(
    x = co_finalDF['Date'],
    y = co_finalDF['Average_CO_Concentration'],
    name = 'CO',
    line_color = 'red',
    opacity = 0.8  
))
co_fig.update_layout(
    xaxis_range = ['1980-01-01', '2019-12-31'], 
    title_text = 'US Daily Avg. CO Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per million)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
co_fig.update_xaxes(automargin = True)
co_fig.update_yaxes(automargin = True)
co_fig.write_image('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures/avg_co.png')

# Split the data into the train/test sets based on the date
co_mask_train = (co_finalDF['Date'] < '2019-01-01')
co_mask_test = (co_finalDF['Date'] >= '2019-01-01')
co_train, co_test = co_finalDF.loc[co_mask_train], co_finalDF.loc[co_mask_test]

# Set up the Keras TimeSeriesGenerator for converting the data into a form recognizable by the model
ser_train = array(co_train['Average_CO_Concentration'].values)
ser_test = array(co_test['Average_CO_Concentration'].values)
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
co_mod = Sequential([
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
co_mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = co_mod.fit_generator(
    train_gen, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Show a summary of the model
print("Summary of the model: %s\n" % co_mod.summary())

# Creating the folder to save models in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models')
    
# Save the model as an HDF5 object (as an .h5 file)
path = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models/co_model.h5'
co_mod.save(path, overwrite = True)

# Make a test prediction
x_in = array(co_test['Average_CO_Concentration'].head(n_in)).reshape((1, n_in, n_feat))
co_pred = co_mod.predict(x_in, verbose = 0)
print('\nPredicted daily avg. CO concentration: %.3f parts per million\n' % co_pred[0][0])
print(co_finalDF[co_finalDF['Date'] == '2019-01-03'])

# Creating the folder to save matplotlib figures in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures')

# Plotting the model's performance metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('CO Data LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
plt.legend()
plt.savefig('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures/co_modelmetrics.png')
