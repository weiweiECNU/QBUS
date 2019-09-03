# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pylab import rcParams
from datetime import datetime



#%%

dateparse = lambda dates: pd.datetime.strptime(dates, '%d/%m/%Y')
data = pd.read_csv("CBA_1991-2017.csv", parse_dates=['Date'], index_col='Date',date_parser=dateparse)

f, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(data)

#%%

rolling_data = data.rolling(2,center = True).mean().rolling(12,center = True).mean()
ax1.plot(rolling_data)


#%%


rolling_data12 = data.rolling(12,center = True).mean()
ax2.plot(rolling_data)
ax2.plot(rolling_data12)