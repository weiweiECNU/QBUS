# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#(a)
data_time  = pd.read_csv("Airline.csv",parse_dates=[['Month', 'Year']],
                         index_col = "Month_Year" ) 

print(data_time.describe())

plt.figure()
plt.plot(data_time)

data = pd.read_csv("Airline.csv") 
passenger = data["Passengers"]
#(b)

#the time series y, 
#the smoothing parameter for level α 
#and the smoothing parameter for the trend β, 
#and returns the smoothed time series.
def holt(y, alpha, beta):
    
    l = [ y[:12].mean() ]
    b = [  (y[12:24]-y[:12])/144 ]
    holtsmoothed = [ l[0] + b[0] ]
    
    Y = y.tolist()
    
    for i in range(len(Y)+4):
        if i == len(Y) - 1:
            Y.append(l[-1] + b[-1])
            
        l.append(alpha * Y[i+1] + (1 - alpha) * (l[i] + b[i]))
        b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])
        holtsmoothed.append(l[i+1] + b[i+1])
    
    return holtsmoothed

alphas = [ 0.2, 0.4, 0.6, 0.8]
betas = [0.2, 0.4, 0.6, 0.8]



a = [1,2,3,4]
