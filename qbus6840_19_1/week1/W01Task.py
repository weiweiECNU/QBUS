#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:39:48 2019

@author: kubbistar
"""

import pandas as pd
import numpy as np

#%%
#task 1
names = pd.Series(["david", "jess", "mark", "laura"])
for name in names:
    if name != "mark":
        print(name)
        
#%%
#task 2

dateparse1 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
sale_data_time1 = pd.read_csv('parse_ex1.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse1)
print(sale_data_time1.head())

#%%
dateparse2 = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
sale_data_time2 = pd.read_csv('parse_ex2.csv', parse_dates=['Time'], index_col='Time',date_parser=dateparse2)
print(sale_data_time2.head())
