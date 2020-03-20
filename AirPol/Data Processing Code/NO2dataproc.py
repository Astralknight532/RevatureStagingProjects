# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:48:16 2020

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
no2_datadir = r'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Raw Data'
no2_df = pd.DataFrame()
for f in os.listdir(no2_datadir):
    if f.endswith(".csv"):
        no2_df = no2_df.append(pd.read_csv(os.path.join(no2_datadir, f), parse_dates = ['Date Local'], infer_datetime_format = True, squeeze = True, usecols = ['Date Local', 'Arithmetic Mean'], encoding = 'utf-8-sig', low_memory = False)[['Date Local', 'Arithmetic Mean']], ignore_index = True)

# Get info about the data
#print("Info about the data(raw): \n%s\n" % no2_df.info())
#print("The first 5 rows of the NO2 data(raw):\n%s\n" % no2_df.head())
#print("The last 5 rows of the NO2 data(raw):\n%s" % no2_df.tail())

# Perform data cleaning on the Pandas dataframes as needed
no2_df.sort_values(by = ['Date Local'], ascending = True, inplace = True, kind = 'mergesort', ignore_index = True) # Sort the rows by date in ascending order
#no2_df = no2_df.drop_duplicates('Date Local') # Drop duplicate rows/entries in the data - reconsider since there are multiple readings per day (i.e. different locations, same date)
#for c in no2_df['Arithmetic Mean'].values: # Fill in null values with the mean of the data
#    no2_df['Arithmetic Mean'] = no2_df['Arithmetic Mean'].fillna(no2_df['Arithmetic Mean'].mean())
#print(no2_df[no2_df['Date Local'] == '1980-01-01'].count())
#print(no2_df[no2_df['Date Local'] == '1980-01-01'].mean())
#print("Info about the data(sorted): \n%s\n" % no2_df.info())
#print("The first 5 rows of the NO2 data(sorted):\n%s\n" % no2_df.head())
#print("The last 5 rows of the NO2 data(sorted):\n%s" % no2_df.tail())

# Creating a new dataframe with the average daily concentration of NO2 and the corresponding date
# Calculate the mean for each day
no2_means = []
for u in no2_df['Date Local'].unique():
    no2_means.append(no2_df[no2_df['Date Local'] == u].mean())

# Setting up a dictionary containing the new data
no2_rd = {'Date': no2_df['Date Local'].unique(), 'Average_NO2_Concentration': no2_means}    

# Converting the dictionary into a pandas Dataframe
no2_finalDF = pd.DataFrame(no2_rd, columns = ['Date', 'Average_NO2_Concentration'])
for d in no2_finalDF['Date']: # Setting the date format to YYYY-MM-DD
    no2_finalDF.loc[no2_finalDF.Date == d, 'Date'] = d.strftime('%Y-%m-%d')
no2_finalDF['Average_NO2_Concentration'] = no2_finalDF['Average_NO2_Concentration'].astype(str) # Converting the values of this column into strings (for use with regex)   
no2_finalDF['Average_NO2_Concentration'] = no2_finalDF['Average_NO2_Concentration'].str.extract('(\d*\.\d*)').astype('float64') # Extracting only the number from the string and converting it to a float  
#print("Info about the data(clean): \n%s\n" % no2_finalDF.info())
#print("The first 5 rows of the NO2 data(clean):\n%s\n" % no2_finalDF.head())
#print("The last 5 rows of the NO2 data(clean):\n%s" % no2_finalDF.tail())

# Checking for the folder to store the cleaned data in & creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Clean Data'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Clean Data')

# Saving the cleaned data as a CSV file
cleaned_no2csv = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Clean Data/cleaned_NO2Data.csv'
no2_finalDF.to_csv(cleaned_no2csv, date_format = '%Y-%m-%d')

# Checking for the folder that plotly figures will be saved in, creating it if it doesn't exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures')

'''    
# Plotting the data used to train the model (the daily average concentration of NO2 in PPB)
no2fig = px.scatter(no2_finalDF, x = 'Date', y = 'Average_NO2_Concentration', width = 3000, height = 2500)
no2fig.add_trace(go.Scatter(
    x = no2_finalDF['Date'],
    y = no2_finalDF['Average_NO2_Concentration'],
    name = 'NO2',
    line_color = 'red',
    opacity = 0.8  
))
no2fig.update_layout(
    xaxis_range = ['1980-01-01', '2019-12-31'], 
    title_text = 'US Daily Avg. NO2 Concentration',
    xaxis = go.layout.XAxis(title_text = 'Date'),
    yaxis = go.layout.YAxis(title_text = 'Daily Avg. Concentration (parts per billion)'),
    font = dict(
        family = 'Courier New, monospace',
        size = 24
    )
)
no2fig.update_xaxes(automargin = True)
no2fig.update_yaxes(automargin = True)
no2fig.write_image('C:/Users/hanan/Desktop/StagingProjects/AirPol/plotly figures/avg_no2.png')
'''

# Split the data into the train/test sets based on the date
no2mask_train = (no2_finalDF['Date'] < '2019-01-01')
no2mask_test = (no2_finalDF['Date'] >= '2019-01-01')
no2train, no2test = no2_finalDF.loc[no2mask_train], no2_finalDF.loc[no2mask_test]

# Set up the Keras TimeSeriesGenerator for converting the data into a form recognizable by the model
ser_train = array(no2train['Average_NO2_Concentration'].values)
ser_test = array(no2test['Average_NO2_Concentration'].values)
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
no2mod = Sequential([
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
no2mod.compile(optimizer = opt, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])
history = no2mod.fit_generator(
    train_gen, 
    steps_per_epoch = 10, 
    epochs = 500,
    verbose = 0
)

# Show a summary of the model
print("Summary of the model: %s\n" % no2mod.summary())

# Creating the folder to save models in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models')
    
# Save the model as an HDF5 object (as an .h5 file)
path = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/Saved Models/no2_model.h5'
no2mod.save(path, overwrite = True)

# Make a test prediction
x_in = array(no2test['Average_NO2_Concentration'].head(n_in)).reshape((1, n_in, n_feat))
no2pred = no2mod.predict(x_in, verbose = 0)
print('\nPredicted daily avg. NO2 concentration: %.3f parts per billion\n' % no2pred[0][0])
print(no2_finalDF[no2_finalDF['Date'] == '2019-01-03'])

# Creating the folder to save matplotlib figures in if it doesn't already exist
if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures'):
    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures')

# Plotting the model's performance metrics
plt.rcParams['figure.figsize'] = (20, 10)
plt.title('NO2 Data LSTM Model Metrics')
plt.xlabel('Epochs')
plt.ylabel('Model Error')
plt.plot(history.history['mse'], label = 'MSE', color = 'red')
plt.plot(history.history['loss'], label = 'MSLE', color = 'blue')
plt.legend()
plt.savefig('C:/Users/hanan/Desktop/StagingProjects/AirPol/matplotlib figures/no2modelmetrics.png')