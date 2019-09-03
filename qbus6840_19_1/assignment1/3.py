# -*- coding: utf-8 -*-
#the dataset plastic.csv which consists of the monthly sales (in thousands)
#of product A for a plastics manufacturer for fives years. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('plastics.csv',index_col = 'Month')

# a
plt.figure()

plt.plot(data)

#b

plastic  = data

Trend = plastic.rolling(2, center = True).mean().rolling(12,center = True).mean().shift(-1)



res_seasonal  = plastic / Trend
#
res_seasonal_zero = np.nan_to_num(res_seasonal)
#
monthly_S = np.reshape(res_seasonal_zero, (5,12))
#
monthly_avg = np.mean(monthly_S[1:4,:], axis=0)
#
mean_allmonth = monthly_avg.mean()
#
monthly_avg_normalized = monthly_avg / mean_allmonth
#
tiled_avg = np.tile(monthly_avg_normalized, 5)
#
#
plt.figure()
#
plt.plot(plastic, label = "plastic")
plt.plot(Trend, label = "trend")
plt.legend()

plt.figure()
plt.plot(tiled_avg, label = "seasonal")
plt.legend()
#
#
#
from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
#Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used']) 
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value 
    print(dfoutput)
                            
                            
                            
def plot_curve(timeseries): #Determing rolling statistics
# rolmean = pd.rolling_mean(pd.rolling_mean(timeseries, window=12),window=2)
# rolstd = pd.rolling_std(pd.rolling_std(timeseries, window=12),window=2)
    rolmean = timeseries.rolling( window=12).mean()
    rolstd = timeseries.rolling( window=12).std() #Plot rolling statistics:
    plt.figure()
    plt.plot(timeseries, color='blue',label='Original') 
    plt.plot(rolmean, color='red', label='Rolling Mean') 
    plt.plot(rolstd, color='black', label = 'Rolling Std') 
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation') 
    plt.show()
#    
# 
test_stationarity(Trend.dropna().iloc[:,0])
#c

seasonally_adjusted = plastic['x'] / tiled_avg


plt.figure()
plt.title("seasonally_adjusted")
plt.plot(seasonally_adjusted)

#d 
outliner = 40
plastic.iloc[outliner,0] += 500


Trend2 = plastic.rolling(2, center = True).mean().rolling(12,center = True).mean().shift(-1)

res_seasonal2  = plastic / Trend2

res_seasonal_zero2 = np.nan_to_num(res_seasonal2)

monthly_S2 = np.reshape(res_seasonal_zero2, (5,12))

monthly_avg2 = np.mean(monthly_S2[1:4,:], axis=0)

mean_allmonth2 = monthly_avg2.mean()

monthly_avg_normalized2 = monthly_avg2 / mean_allmonth2

tiled_avg2 = np.tile(monthly_avg_normalized2, 5)

seasonally_adjusted2 = plastic['x'] / tiled_avg2

plt.figure()

plt.plot(seasonally_adjusted)
plt.plot(seasonally_adjusted2)
plt.legend(["Oringinal","outliner"])
#
#
# e
plastic.iloc[outliner,0] -= 500

res = res_seasonal.dropna()
test_stationarity(res.iloc[:,0])
plot_curve(res.iloc[:,0])

#
#
from sklearn.linear_model import LinearRegression
X = np.arange(1,61)
y = plastic.iloc[:,0]/tiled_avg
#
X = np.reshape(X, (len(plastic), 1))
y = y.values.reshape(len(plastic),1)
#    
lm = LinearRegression()
lm.fit(X, y)
#    
# The coefficients
print("Coefficients: {0}".format(lm.coef_))
# The intercept
print("Intercept: {0}".format(lm.intercept_))
print("Total model: y = {0} + {1} X".format(lm.intercept_,lm.coef_[0]))

print("Variance score (R^2): {0:.2f}".format(lm.score(X, y)))
#
#
#
trend_cycle = lm.predict(np.reshape(np.arange(1,61), (60,1)))

linear_seasonal_total = trend_cycle * np.reshape(tiled_avg, (60,1))
#
plt.figure()
plt.plot(X, y, label = "reestimated_seasonally_adjusted")
plt.plot(trend_cycle, linestyle='--', label="Forecast_trend")
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(X,plastic.iloc[:,0],label = "oringinal data")
plt.plot(X,linear_seasonal_total,label="linear_seasonal_forecast")
plt.legend()
#
#
tiled_avg_forcast = np.tile(monthly_avg_normalized, 6)
trend_cycle_f = lm.predict(np.reshape(np.arange(1,73), (72,1)))
linear_seasonal_forecast = trend_cycle_f * np.reshape(tiled_avg_forcast, (72,1))
print(linear_seasonal_forecast[61,0])
print(linear_seasonal_forecast[62,0])
print(linear_seasonal_forecast[63,0])