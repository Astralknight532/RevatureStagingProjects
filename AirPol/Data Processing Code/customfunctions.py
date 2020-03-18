# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:39:32 2020

@author: hanan
"""

import pandas as pd
import math as m

# Defining functions to organize repetitive tasks

# Converts date column to datetime
def dt_convert(data:pd.Series) -> pd.Series: 
    data = pd.to_datetime(data.str.strip(), format = '%Y-%m-%d', errors = 'ignore')
    return data

# Converts the passed string values into type float64 - extracts strings using regex
def float_convert(data:pd.Series) -> pd.Series:
    data = data.str.extract('(\d*\.\d*)').astype('float64')
    return data

# Converting between degrees and radians (used in the function lat_long_dist)
def d_to_r(deg): # degrees to radians
    return (deg * m.pi) / 180
def r_to_d(rad): # radians to degrees
    return (rad * 180) / m.pi

# Distance between 2 latitude-longitude points
def lat_long_dist(lat1, long1, lat2, long2):
    earth_r = 6378
    lat_dif = d_to_r(lat2 - lat1)
    long_dif = d_to_r(long2 - long1)
    s1 = (m.sin(lat_dif / 2) ** 2) + (m.cos(d_to_r(lat1)) * m.cos(d_to_r(lat2))) * (m.sin(long_dif / 2) ** 2)
    s2 = 2 * m.atan2(m.sqrt(s1), m.sqrt(1 - s1))
    return earth_r * s2