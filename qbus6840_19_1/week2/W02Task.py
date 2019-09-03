#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:46:17 2019

@author: boyanzhang
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

#%%
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d-%H')
data = pd.read_csv('BeijingPM20100101_20151231.csv', parse_dates=['year-month-day-hour'], index_col='year-month-day-hour',date_parser=dateparse)
print(data.head())
PM_US_Post = data["PM_US Post"]

#%%
PM_US_Post_2011 =  PM_US_Post["2011-03"]
plt.figure()
plt.plot(PM_US_Post_2011)
plt.title('PM2.5 concentration 2011-03')
plt.xlabel('day')
plt.ylabel('ug/m^3')
mean_PM_2011 = np.mean(PM_US_Post_2011)
#%%
PM_US_Post_2012 =  PM_US_Post["2012-03"]
plt.figure()
plt.plot(PM_US_Post_2012)
plt.title('PM2.5 concentration 2012-03')
plt.xlabel('day')
plt.ylabel('ug/m^3')
mean_PM_2012 = np.mean(PM_US_Post_2012)
#%%
PM_US_Post_2013 =  PM_US_Post["2013-03"]
plt.figure()
plt.plot(PM_US_Post_2013)
plt.title('PM2.5 concentration 2013-03')
plt.xlabel('day')
plt.ylabel('ug/m^3')
mean_PM_2013 = np.mean(PM_US_Post_2013)
#%%
PM_US_Post_2014 =  PM_US_Post["2014-03"]
plt.figure()
plt.plot(PM_US_Post_2014)
plt.title('PM2.5 concentration 2014-03')
plt.xlabel('day')
plt.ylabel('ug/m^3')
mean_PM_2014 = np.mean(PM_US_Post_2014)
#%%
PM_US_Post_2015 =  PM_US_Post["2015-03"]
plt.figure()
plt.plot(PM_US_Post_2015)
plt.title('PM2.5 concentration 2015-03')
plt.xlabel('day')
plt.ylabel('ug/m^3')
mean_PM_2015 = np.mean(PM_US_Post_2015)

#%%
index = pd.DatetimeIndex(['2011', '2012', '2013', '2014', '2015'])
PM_03_data = pd.Series([mean_PM_2011,mean_PM_2012,mean_PM_2013,mean_PM_2014,mean_PM_2015], index=index)
plt.figure()
plt.plot(PM_03_data)
