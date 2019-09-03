import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.pylab import rcParams
from datetime import datetime

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data_time = pd.read_csv("AirPassengers.csv", parse_dates=['Month'], index_col='Month',date_parser=dateparse)

ts = data_time["Passengers"]

rcParams['figure.figsize'] = 15, 6

plt.figure()
plt.plot(ts)


s =pd.Series( [1,2,3,5,6,10,12,14,12,30])
#.rolling(window=3,center=True).mean()

rolling_data = ts.rolling(12,center=True)
plt.figure()
plt.plot(ts,'r-',label="noised data")
plt.plot(rolling_data.mean(),'b-',label="Rolling mean")
plt.plot(rolling_data.std(),'g-',label="Rolling std")
plt.legend()

ts_log= np.log(ts)
plt.figure()
plt.plot(ts_log, color='red',label='log')

Trend = ts_log.rolling(2, center =True).mean().rolling(12,center = True).mean()

plt.figure()
plt.plot(ts_log, color='red',label='log')
plt.plot(Trend, color='blue',label='MA')
plt.title('Initial TREND estimate ')
plt.xlabel('Month')
plt.ylabel('Number')

t1 = pd.Series([1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4])



t2 = pd.Series([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])

ts_res = ts_log - Trend
ts_res.dropna(inplace = True)

rolling_data_res = pd.Series(ts_res,
dtype='float64').rolling(12,center=True)
plt.figure()
plt.plot(ts_res,'r-',label="time series data")
plt.plot(rolling_data_res.mean(),'b-',label="Rolling mean")
plt.plot(rolling_data_res.std(),'g-',label="Rolling std")
plt.legend()


from statsmodels.tsa.stattools import adfuller



def test_stationarity(timeseries):
#Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']) 
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value 
    print(dfoutput)
        
    
def plot_curve(timeseries): 
    
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    plt.figure()
    plt.plot(timeseries, color='blue',label='Original') 
    plt.plot(rolmean, color='red', label='Rolling Mean') 
    plt.plot(rolstd, color='black', label = 'Rolling Std') 
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation') 
    plt.show()
    
    
plot_curve(ts)   
test_stationarity(ts)

plot_curve(ts_res)
test_stationarity(ts_res)




