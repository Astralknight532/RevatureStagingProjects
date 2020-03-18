# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:39:32 2020

@author: hanan
"""

import pandas as pd
from math import sin, cos, sqrt, atan2, radians, pi

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
    return (deg * pi) / 180
def r_to_d(rad): # radians to degrees
    return (rad * 180) / pi

# Distance between 2 latitude-longitude points in kilometers
def lat_long_dist(lat1, long1, lat2, long2):
    earth_r = 6378 # radius of the Earth in kilometers
    lat_dif = radians(lat2) - radians(lat1)
    long_dif = radians(long2) - radians(long1)
    s1 = (sin(lat_dif / 2) ** 2) + (cos(radians(lat1)) * cos(radians(lat2))) * (sin(long_dif / 2) ** 2)
    s2 = 2 * atan2(sqrt(s1), sqrt(1 - s1))
    return earth_r * s2