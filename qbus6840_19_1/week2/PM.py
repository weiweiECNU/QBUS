import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d-%H')
data_time= pd.read_csv('BeijingPM20100101_20151231.csv', parse_dates=['year-month-day-hour'], index_col='year-month-day-hour',date_parser=dateparse)

pm_us = data_time["PM_US Post"]

years = [ i.strftime("%Y") for i in pd.date_range(start='2010', periods=6, freq='Y') ]

pm2010 = pm_us['2010']
pm2011 = pm_us['2011']
pm2012 = pm_us['2012']
pm2013 = pm_us['2013']
pm2014 = pm_us['2014']
pm2015 = pm_us['2015']


pm2010march = pm_us['2010-03']
pm2011march = pm_us['2011-03']
pm2012march = pm_us['2012-03']
pm2013march = pm_us['2013-03']
pm2014march = pm_us['2014-03']
pm2015march = pm_us['2015-03']


marches = [ i.strftime("%Y-%m") for i in pd.date_range(start='2010-03', periods=6, freq='12M') ]

#pm_data = []
#
#for i in marches:
#    pm_data.append(pm_us[i].mean())
#    

pm_data = [ m.mean() for m in [pm_us[i] for i in marches ] ]
plt.plot(  years ,pm_data)