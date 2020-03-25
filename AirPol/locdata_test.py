# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:26:06 2020

@author: hanan
"""

import folium
import os
import pandas as pd
import json
#from folium import plugins

# Read in the data (multiple CSV files, 1 file per year of data from 1980 to 2019)
no2_datadir = r'C:/Users/hanan/Desktop/StagingProjects/AirPol/USAirPolData/NO2 Data/Raw Data'
no2_df = pd.DataFrame()
for f in os.listdir(no2_datadir):
    if f.endswith(".csv"):
        no2_df = no2_df.append(pd.read_csv(os.path.join(no2_datadir, f), parse_dates = ['Date Local'], infer_datetime_format = True, squeeze = True, usecols = ['Date Local', 'Arithmetic Mean', 'Latitude', 'Longitude'], encoding = 'utf-8-sig', low_memory = False)[['Date Local', 'Arithmetic Mean', 'Latitude', 'Longitude']], ignore_index = True)
        
# Perform data cleaning on the Pandas dataframes as needed
no2_df.sort_values(by = ['Date Local'], ascending = True, inplace = True, kind = 'mergesort', ignore_index = True) # Sort the rows by date in ascending order

# Read in the Geojson file that will be used as a base for the map
with open("C:/Users/hanan/Desktop/StagingProjects/AirPol/US2010_500k_states.json") as f:
    us_base = json.load(f)

# Creating a simple point map
us_map = folium.Map(location = [48, -102], zoom_start = 3)
folium.GeoJson(us_base).add_to(us_map)

for i, row in no2_df.iterrows():
    folium.CircleMarker((row.Latitude, row.Longitude), radius = 3, weight = 2, color = 'red', fill_color = 'red', fill_opacity = 0.5).add_to(us_map)
   
# Saving the map as a PNG image
us_map.save("C:/Users/hanan/Desktop/StagingProjects/AirPol/usmap_NO2monitor.png")    
