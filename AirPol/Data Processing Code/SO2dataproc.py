# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:50:44 2020

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
so2_datadir = r'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/SO2 Data/Raw Data'
so2_df = pd.DataFrame()
for f in os.listdir(so2_datadir):
    if f.endswith(".csv"):
        so2_df = so2_df.append(pd.read_csv(os.path.join(so2_datadir, f), parse_dates = ['Date Local'], infer_datetime_format = True, squeeze = True, usecols = ['Date Local', 'Arithmetic Mean'], encoding = 'utf-8-sig', low_memory = False)[['Date Local', 'Arithmetic Mean']], ignore_index = True)

# Get info about the data
#print("Info about the data(raw): \n%s\n" % so2_df.info())
#print("The first 5 rows of the SO2 data(raw):\n%s\n" % so2_df.head())
#print("The last 5 rows of the SO2 data(raw):\n%s\n" % so2_df.tail())
        
# Perform data cleaning on the Pandas dataframes as needed
so2_df.sort_values(by = ['Date Local'], ascending = True, inplace = True, kind = 'mergesort', ignore_index = True) # Sort the rows by date in ascending order
#print("Info about the data(sorted): \n%s\n" % so2_df.info())
#print("The first 5 rows of the SO2 data(sorted): \n%s\n" % so2_df.head())
#print("The last 5 rows of the SO2 data(sorted): \n%s" % so2_df.tail())

# Creating a new dataframe with the average daily concentration of SO2 and the corresponding date
# Calculate the mean for each day
so2_means = []
for u in so2_df['Date Local'].unique():
    so2_means.append(so2_df[so2_df['Date Local'] == u].mean())

# Setting up a dictionary containing the new data
so2_rd = {'Date': so2_df['Date Local'].unique(), 'Average_SO2_Concentration': so2_means}    

# Converting the dictionary into a pandas Dataframe
so2_finalDF = pd.DataFrame(so2_rd, columns = ['Date', 'Average_SO2_Concentration'])
for d in so2_finalDF['Date']: # Setting the date format to YYYY-MM-DD
    so2_finalDF.loc[so2_finalDF.Date == d, 'Date'] = d.strftime('%Y-%m-%d')
so2_finalDF['Average_SO2_Concentration'] = so2_finalDF['Average_SO2_Concentration'].astype(str) # Converting the values of this column into strings (for use with regex)   
so2_finalDF['Average_SO2_Concentration'] = so2_finalDF['Average_SO2_Concentration'].str.extract('(\d*\.\d*)').astype('float64') # Extracting only the number from the string and converting it to a float  
#print("Info about the data(clean): \n%s\n" % so2_finalDF.info())
#print("The first 5 rows of the SO2 data(clean):\n%s\n" % so2_finalDF.head())
#print("The last 5 rows of the SO2 data(clean):\n%s" % so2_finalDF.tail())

# Checking for the folder to store the cleaned data in & creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/SO2 Data/Clean Data'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/SO2 Data/Clean Data')

# Saving the cleaned data as a CSV file
cleaned_so2csv = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/SO2 Data/Clean Data/cleaned_SO2Data.csv'
so2_finalDF.to_csv(cleaned_so2csv, date_format = '%Y-%m-%d')

# Checking for the folder that plotly figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures')

''' 
# Plotting the data used to train the model (the daily average concentration of SO2 in PPB)
so2fig = px.scatter(so2_finalDF, x = 'Date', y = 'Average_SO2_Concentration', width = 3000, height = 2500)
so2fig.add_trace(go.Scatter(
    x = so2_finalDF['Date'],
    y = so2_finalDF['Average_SO2_Concentration'],
    name = 'SO2',
    line_color = 'red',
    opacity = 0.8  
))
so2fig.update_layout(
    xaxis_range = ['1980-01-01', '2019-12-31'], 
    title_text = 'US Daily Avg. SO2 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per billion)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
so2fig.update_xaxes(automargin = True)
so2fig.update_yaxes(automargin = True)
so2fig.write_image('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures/avg_so2.png')
'''

# Split the data into the train/test sets based on the date
so2mask_train = (so2_finalDF['Date'] < '2019-01-01')
so2mask_test = (so2_finalDF['Date'] >= '2019-01-01')
so2train, so2test = so2_finalDF.loc[so2mask_train], so2_finalDF.loc[so2mask_test]

# Set up the Keras TimeSeriesGenerator for converting the data into a form recognizable by the model
ser_train = array(so2train['Average_SO2_Concentration'].values)
ser_test = array(so2test['Average_SO2_Concentration'].values)
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
so2mod = Sequential([
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
so2mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = so2mod.fit_generator(
    train_gen, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Show a summary of the model
print("Summary of the model: %s\n" % so2mod.summary())

# Creating the folder to save models in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models')
    
# Save the model as an HDF5 object (as an .h5 file)
path = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models/so2_model.h5'
so2mod.save(path, overwrite = True)

# Make a test prediction
x_in = array(so2test['Average_SO2_Concentration'].head(n_in)).reshape((1, n_in, n_feat))
so2pred = so2mod.predict(x_in, verbose = 0)
print('\nPredicted daily avg. SO2 concentration: %.3f parts per billion\n' % so2pred[0][0])
print(so2_finalDF[so2_finalDF['Date'] == '2019-01-03'])

# Creating the folder to save matplotlib figures in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures')

# Plotting the model's performance metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('SO2 Data LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
plt.legend()
plt.savefig('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures/so2modelmetrics.png')