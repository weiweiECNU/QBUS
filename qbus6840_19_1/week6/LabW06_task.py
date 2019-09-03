#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:16:11 2019

@author: boyanzhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:04:05 2017

@author: Boyan Zhang
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

visitors = pd.read_csv('AustralianVisitors.csv')
# The original number is too large to good visualization,
# In here, I resize the data by dividing 10000
y = visitors['No of Visitors']/10000

#%% 
def sse(x, y):
    return np.sum(np.power(x - y,2))

def trend_smooth(alpha,beta,y):
    l = [y[0]]
    b = [y[1] - y[0]]
    holtsmoothed_manual = [l[0] + b[0]]

    Y = y.tolist()
    for i in range(len(y)):
        if i == len(Y) - 1:   
            Y.append(l[-1] + b[-1])
        l.append(alpha * Y[i+1] + (1 - alpha) * (l[i] + b[i])) 
        b.append(beta * (l[i+1] - l[i]) + (1 - beta) * b[i])
        holtsmoothed_manual.append(l[i+1] + b[i+1])
    
    # the len of holtsmoothed_manual should be t+1
    # we do a forecasting for T = t+1 in conveinence of calculation
    # However, if you only focus on smoothing result, then you should setting holtsmoothed_manual[:-1]
    # 
    # For resdual calculation of the one-step forecast, 
    # make sure use y2-y^2, y3-y^3, ..., yt - y^t
    # that is holtsmoothed_manual[1:-1] for y^2:t
    # and y.values[1:] for y2:t
    return holtsmoothed_manual[:-1], sse(holtsmoothed_manual[1:-1], y.values[1:]) 


#%%
alphas = np.arange(0.,1,0.1)
betas = np.arange(0.,1,0.1)

alpha_fit = 0
beta_fit = 0
smoothed_fit = []

# initialize a inf number 
min_res = float('inf')

# go through each alpha and beta
for alpha in alphas:
    for beta in betas:
        smoothed_temp , res_temp = trend_smooth(alpha, beta, y)
        # if the current resdual is smaller than the min of the previous resdual
        # then we save the current alpha,beta,smoothing results and current min resdual
        if res_temp <= min_res:          
            alpha_fit = alpha
            beta_fit = beta
            smoothed_fit = smoothed_temp
            min_res = res_temp

plt.figure()
plt.plot(smoothed_fit,label ="smoothing")
plt.plot(y,label="observation")
plt.legend(loc="upper left")
plt.title("Final Holt's trend smoothing result, alpha ={0}, beta = {1}".format(alpha,beta))
        
        
        
        