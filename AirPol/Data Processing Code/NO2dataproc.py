# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:48:16 2020

@author: hanan
"""

# Import needed libraries
#import customfunctions as cf # a Python file with functions I wrote for repetitive tasks
import os
import pandas as pd
#from time import strftime
#from numpy import array
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#import plotly.express as px
#from keras.preprocessing.sequence import TimeseriesGenerator
#from keras.optimizers import SGD
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout

# Read in the data (multiple CSV files, 1 file per year of data from the 1980s to 2019)
no2_datadir = r'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Raw Data'
no2_df = pd.DataFrame()
for f in os.listdir(no2_datadir):
    if f.endswith(".csv"):
        no2_df = no2_df.append(pd.read_csv(os.path.join(no2_datadir, f), parse_dates = ['Date Local'], infer_datetime_format = True, squeeze = True, usecols = ['Date Local', 'Arithmetic Mean'], encoding = 'utf-8-sig', low_memory = False)[['Date Local', 'Arithmetic Mean']], ignore_index = True)

# Get info about the data
#print(no2_df.info())
#print(no2_df)
#print("The first 5 rows of the NO2 data:\n%s\n" % no2_df.head())
#print("The last 5 rows of the NO2 data:\n%s" % no2_df.tail())
#mask = (no2_df['Date Local'] > '1980-12-31') & (no2_df['Date Local'] < '2019-01-01')
#print(no2_df.loc[mask])

# Perform data cleaning on the Pandas dataframes as needed (there are issues with the sorting, do not use it)
no2_df.sort_values(by = ['Date Local'], ascending = True, inplace = True, kind = 'mergesort', ignore_index = True) # Sort the rows by date in ascending order
#no2_df = no2_df.drop_duplicates('Date Local') # Drop duplicate rows/entries in the data - reconsider since there are multiple readings per day (i.e. different locations, same date)
#for c in no2_df['Arithmetic Mean'].values: # Fill in null values with the mean of the data
#    no2_df['Arithmetic Mean'] = no2_df['Arithmetic Mean'].fillna(no2_df['Arithmetic Mean'].mean())
#print(no2_df[no2_df['Date Local'] == '1980-01-01'].count())
#print(no2_df[no2_df['Date Local'] == '1980-01-01'].mean())

# Creating a new dataframe with the average daily concentration of NO2
# Calculate the mean for each day
no2_means = []
for u in no2_df['Date Local'].unique():
    no2_means.append(no2_df[no2_df['Date Local'] == u].mean())
#print(no2_means)

# Setting up a dictionary containing the new data
no2_rd = {'Date': no2_df['Date Local'].unique(), 'Average_NO2_Concentration': no2_means}    

# Converting the dictionary into a pandas Dataframe
no2_finalDF = pd.DataFrame(no2_rd, columns = ['Date', 'Average_NO2_Concentration'])
for d in no2_finalDF['Date']: # Setting the date format to YYYY-MM-DD
    no2_finalDF.loc[no2_finalDF.Date == d, 'Date'] = d.strftime('%Y-%m-%d')
no2_finalDF['Average_NO2_Concentration'] = no2_finalDF['Average_NO2_Concentration'].astype(str) # Converting the values of this column into strings (for use with regex)   
no2_finalDF['Average_NO2_Concentration'] = no2_finalDF['Average_NO2_Concentration'].str.extract('(\d*\.\d*)').astype('float64') # Extracting only the number from the string and converting it to a float  

# Checking for the folder to store the cleaned data in & creating it if it doesn't exist
#if not os.path.exists('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Clean Data'):
#    os.mkdir('C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Clean Data')

# Saving the cleaned data as a CSV file in the folder above
#print(no2_finalDF.info())
#print("The first 5 rows of the NO2 data:\n%s\n" % no2_finalDF.head())
#print("The last 5 rows of the NO2 data:\n%s" % no2_finalDF.tail())
cleaned_no2csv = 'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Clean Data/cleaned_NO2Data.csv'
no2_finalDF.to_csv(cleaned_no2csv, date_format = '%Y-%m-%d')

# Plotting the data used to train the model

# Split the data into the train/test sets based on the date

# Set up the Keras TimeSeriesGenerator for converting the data into a form recognizable by the model

# Defining an alternate optimizer (in case the ADAM optimizer doesn't work well for this application)
#opt = SGD(lr = 0.01, momentum = 0.9, nesterov = True)

# Defining the model's structure

# Compiling & fitting the model

# Show a summary of the model

# Save the model as an HDF5 object (as an .h5 file)

# Make a test prediction

# Plotting the model's performance metrics