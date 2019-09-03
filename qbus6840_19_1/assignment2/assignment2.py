# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

date_parser = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d") 

data_daily  = pd.read_csv("ASX200Daily.csv",parse_dates = ["Date"], 
                         index_col = 'Date', date_parser = date_parser) 

data_monthly = pd.read_csv("ASX200Monthly.csv",parse_dates = ["Date"], 
                         index_col = 'Date', date_parser = date_parser)

close_daily = data_daily["Close"]
close_monthly = data_monthly["Close"]

close_daily.fillna(method='ffill', inplace=True)
close_monthly.fillna(method='ffill', inplace=True)
close_monthly.dropna(inplace=True)

plt.figure(figsize=(16,5))
plt.plot(close_daily)
plt.grid()
plt.title("Daily close")
plt.xlabel("Time")
plt.ylabel("Close")

plt.figure(figsize=(16,5))
plt.plot(close_monthly)
plt.grid()
plt.title("Monthly close")
plt.xlabel("Time")
plt.ylabel("Close")

diff_close_daily = pd.Series.diff(close_daily)
diff_close_monthly = pd.Series.diff(close_monthly)
diff_close_daily.dropna(inplace = True)
diff_close_monthly.dropna(inplace = True)

plt.figure(figsize = (16,5))
plt.plot(diff_close_daily)
plt.title("1 step difference Daily close")
plt.xlabel("Time")
plt.ylabel("Difference")


plt.figure(figsize = (16,5))
plt.plot(diff_close_monthly)
plt.title("1 step difference monthly close")
plt.xlabel("Time")
plt.ylabel("Difference")


import statsmodels as sm 
import statsmodels.api as smt

smt.graphics.tsa.plot_acf(close_daily, lags = 30, alpha = 0.05); # try diff alpha
smt.graphics.tsa.plot_pacf(close_daily, lags=30, alpha=0.05)
plt.show()


smt.graphics.tsa.plot_acf(diff_close_daily, lags = 30, alpha = 0.05); # try diff alpha
smt.graphics.tsa.plot_pacf(diff_close_daily, lags=30, alpha=0.05)
plt.show()

smt.graphics.tsa.plot_acf(close_monthly, lags = 30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(close_monthly, lags=30, alpha=0.05)
plt.show()

smt.graphics.tsa.plot_acf(diff_close_monthly, lags = 30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(diff_close_monthly, lags=30, alpha=0.05)
plt.show()

print("The description of daily close ",close_daily.describe())
print("The description of monthly close ",close_monthly.describe())


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
close_daily_normalize = scaler.fit_transform(close_daily.values.reshape(-1,1))
close_monthly_normalize = scaler.fit_transform(close_monthly.values.reshape(-1,1))
plt.figure(figsize = (16,5))
plt.plot(close_daily_normalize)
plt.title("MinMaxScaler daily close")
plt.xlabel("Time")
plt.ylabel("Close")


plt.figure(figsize = (16,5))
plt.plot(close_monthly_normalize)
plt.title("MinMaxScaler monthly close")
plt.xlabel("Time")
plt.ylabel("Close")


#############################################
#
#Benchmark Model:
#
##########################################
validation_size = 7
train_size = len(close_daily) - validation_size

train_close_daily = close_daily[:-validation_size]
validation_close_daily  =  close_daily[-validation_size:]

close_daily_log = np.log(close_daily)
train_close_daily_log = np.log(train_close_daily)
validation_close_daily_log = np.log(validation_close_daily)

def sse(x, y): # sse: sum of squared error
    return np.sum(np.power(x - y, 2))


from statsmodels.tsa.holtwinters import Holt
fit1 = Holt(train_close_daily).fit(optimized = True)
smooth_Holt = fit1.fittedvalues

forecast_set = pd.Series(fit1.forecast(validation_size))

forecast_set.index = validation_close_daily.index


plt.figure(figsize = (16,5))
plt.plot(close_daily)
plt.plot(smooth_Holt,linestyle='--')

plt.figure(figsize = (16,5))
plt.plot(close_daily[-50:])
plt.plot(smooth_Holt[-43:])
plt.plot(forecast_set)

validation_set = np.exp(close_daily_log[-validation_size:].values)
print("SSE for Holt’s linear method\n",sse(forecast_set,validation_set))



#####################################
#
#ARIMA
#
#######################################

plt.figure(figsize = (16,5))
plt.plot(close_daily_log)
smt.graphics.tsa.plot_acf(close_daily_log, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(close_daily_log, lags=30, alpha = 0.05)

close_daily_log_diff = close_daily_log.diff()
close_daily_log_diff.dropna(inplace = True)
plt.figure(figsize = (16,5))
plt.plot(close_daily_log_diff)

smt.graphics.tsa.plot_acf(close_daily_log_diff, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(close_daily_log_diff, lags=30, alpha = 0.05)