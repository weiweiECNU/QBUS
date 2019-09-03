# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd


#data = pd.read_csv("AirPassengers.csv")
#
parser = lambda dates : pd.datetime.strptime(dates, "%Y-%m")

#data_time = pd.read_csv("AirPassengers.csv", parse_dates=["Month"], index_col = 'Month', date_parser = parser)

#plt.plot(data_time)


#
#parser = lambda dates: pd.datetime.strptime(dates, "%Y-%m")
#
data_time = pd.read_csv("AirPassengers.csv",parse_dates = ["Month"], index_col = 'Month', date_parser = parser)

a = pd.Series([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4] )
b = pd.Series([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5] )

import numpy as np
x = np.array([1,2,3,4])
y = np.array([4,5,6,7])


import numpy as np
b = np.array([[5,4,3],[0,3,8]]) 
print(b[:,1:-1])