# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

beer_df = pd.read_csv('beer.txt')

X = np.linspace(1, len(beer_df), len(beer_df))
y = beer_df['Sales']
plt.figure()
plt.plot(X, y)
plt.title("Beer Sales")

X = np.reshape(X, (len(beer_df), 1))
#y = np.reshape(y,(len(beer_df),1))
y = y.values.reshape(len(beer_df),1)
lm = LinearRegression()
lm.fit(X, y)

# The coefficients
print("Coefficients: {0}".format(lm.coef_))
# The intercept
print("Intercept: {0}".format(lm.intercept_))

print("Total model: y = {0} + {1} X".format(lm.intercept_,
lm.coef_[0]))

print("Variance score (R^2): {0:.2f}".format(lm.score(X, y)))

trend = lm.predict(X)
plt.figure()
plt.plot(X, y, label = "Beer Sales")
plt.plot(trend, label="Trend")
plt.legend()
plt.title("Beer Sales Trend from Linear Regression Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show(block=False)

sse1 = np.sum( np.power(y - trend,2), axis=0)
sse2 = lm._residues

beer_df['residuals'] = y - trend

acf_vals = [beer_df['residuals'].autocorr(i) for i in range(1,15) ]

plt.figure()
plt.bar(np.arange(1,15), acf_vals)
plt.title("ACF of Beer Sales")
plt.xlabel("Month delay/lag")
plt.ylabel("Correlation Score")
plt.show(block=False)

plt.figure()
plt.title("Residual plot")
plt.scatter(X, trend - y)
plt.xlabel("Month")
plt.ylabel("Residuals")
plt.show(block=False)

forecast = lm.predict(np.reshape(np.arange(72), (72,1)))
plt.figure()
plt.plot(X, y, label="Beer Sales")
plt.plot(trend, label="Trend")
plt.plot(forecast, linestyle='--', label="Forecast")
plt.legend()
plt.title("Beer Sales Forecast from Trend Only Linear Regression Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show(block=False)

T = beer_df.rolling(12, center = True).mean().rolling(2,center = True).mean().shift(-1)

S_additive = beer_df['Sales'] - T['Sales']

safe_S = np.nan_to_num(S_additive)

monthly_S = np.reshape(np.concatenate( (safe_S, [0,0,0,0]),axis = 0), (5,12))

monthly_avg = np.mean(monthly_S[1:4,], axis=0)
mean_allmonth = monthly_avg.mean()
monthly_avg_normed = monthly_avg - mean_allmonth

tiled_avg = np.tile(monthly_avg_normed, 6)

# test the stationality of residual

linear_trend = lm.predict(np.reshape(np.arange(72), (72,1)))

linear_seasonal_forecast = linear_trend + np.reshape(tiled_avg,(72,1))


plt.figure()
plt.plot(X, y, label="Original Data")
plt.plot(linear_trend, label="Linear Model trend")
plt.plot(linear_seasonal_forecast, label="Linear+Seasonal Forecast")
plt.title("Beer Sales Forecast from Trend+Seasonal Linear Regression Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show(block=False)

col = [0,1]
row = [1,2,3,4]
a = pd.DataFrame(columns = col, index = row)


seasons = []
for i in range(y.size):
    if i % 12 == 0:
        seasons = np.append(seasons, 'Jan')
    if i % 12 == 1:
        seasons = np.append(seasons, 'Feb')
    if i % 12 == 2:
        seasons = np.append(seasons, 'Mar')
    if i % 12 == 3:
        seasons = np.append(seasons, 'Apr')
    if i % 12 == 4:
        seasons = np.append(seasons, 'May')
    if i % 12 == 5:
        seasons = np.append(seasons, 'Jun')
    if i % 12 == 6:
        seasons = np.append(seasons, 'Jul')
    if i % 12 == 7:
        seasons = np.append(seasons, 'Aug')
    if i % 12 == 8:
        seasons = np.append(seasons, 'Sep')
    if i % 12 == 9:
        seasons = np.append(seasons, 'Oct')
    if i % 12 == 10:
        seasons = np.append(seasons, 'Nov')
    if i % 12 == 11:
        seasons = np.append(seasons, 'Dec')
        
dummies = pd.get_dummies(seasons, drop_first=True)

dummies = dummies.values

# make sure the size of X and dummies match
# If you are using numpy version below than 1.10, you need to uncomment the following statement
# X = X[:, np.newaxis]
# Now we add these dummy features into feature, stacking along the column
Xnew = np.hstack((X,dummies))

# Create linear regression object (model)
regr = LinearRegression()
# Train the model using the training sets
regr.fit(Xnew, y)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The intercept
print('Intercept: \n', regr.intercept_)

Ypred = regr.predict(Xnew)
plt.plot(y, label='Observed')
plt.plot(Ypred, '-r', label='Predicted')
plt.legend()

