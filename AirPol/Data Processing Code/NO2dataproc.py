# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:48:16 2020

@author: hanan
"""

# Import needed libraries
#import customfunctions as cf # a Python file with functions I wrote for repetitive tasks
#import pandas as pd
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout
#from keras.optimizers import SGD
#from keras.preprocessing.sequence import TimeseriesGenerator
#from numpy import array
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#import plotly.express as px
#import os

# Read in the data (multiple CSV files, 1 file per year of data from the 1980s to 2019)

# Perform data cleaning on the Pandas dataframes as needed

# Checking for the folder to store the cleaned data in & creating it if it doesn't exist

# Saving the cleaned data as a CSV file in the folder above (concatenate the Pandas dataframes beforehand)

# Split the data into the train/test sets based on the date

# Set up the Keras TimeSeriesGenerator for converting the data into a form recognizable by the model

# Defining an alternate optimizer (in case the ADAM optimizer doesn't work well for this application)

# Defining the model's structure

# Compiling & fitting the model

# Show a summary of the model

# Save the model as an HDF5 object (as an .h5 file)

# Make a test prediction

# Plotting the data used to train the model

# Plotting the model's performance metrics