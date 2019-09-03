# -*- coding: utf-8 -*-

#contains the monthly stock prices of Commonwealth Bank of Australia (CBA) 
#from August 1991 to December 2018. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath

# a
date_parser = lambda dates: pd.datetime.strptime(dates, "%d/%m/%y") 

data_time  = pd.read_csv("CBA_1991-2018.csv",parse_dates = ["Date"], 
                         index_col = 'Date', date_parser = date_parser) 

data_high = data_time['High']

data_high.to_csv("CBA_1991-2018High.csv")

high_1_diff = data_high.diff(1)

high_2_diff = high_1_diff.diff(1)

plt.figure()

fig, ax = plt.subplots(3,1)
ax[0].plot(data_high)
ax[1].plot(high_1_diff)
ax[2].plot(high_2_diff)


ax[0].legend(['High'], loc=2)
ax[1].legend(['1 order'], loc=2)
ax[2].legend(['2 order'], loc=2)


# b 

def cma_24(ts):
    result = []
    for index in np.arange( 12 , len(ts)-12 ):
#        print(index)
        re = 0.5*( ts[ index - 12] + ts[ index + 12 ]) + ts[index - 11:index+12].sum() 
        mean = re / 24
        result.append(mean)
    return result

def cma_n(ts,n):
    result = []
    if n % 2 == 0:
        for index in np.arange( n , len(ts)-n ):
#        print(index)
            sum = 0.5*( ts[ index - n] + ts[ index + n ]) + ts[index - (n-1):index+n].sum() 
            mean = sum / 2 / n
            result.append(mean)
        
    nan = [np.nan]*n
    result = nan+result+nan
    return result

#data_12 = cma_n(data_high,12)

#data_12_2 = cma_n(pd.Series(data_12),2)

#data_24 = cma_n(data_high,12)
    
data_cma24_list = cma_24(data_high)
nan = [np.nan]*12
data_cma24_list = nan+data_cma24_list+nan


#data_cma24 = pd.Series(data_12_2, index = data_high.index)

data_cma24 = pd.Series(data_cma24_list, index = data_high.index)
plt.figure()
plt.plot(data_high)
plt.plot(data_cma24)


rolling_data = data_high.rolling(24,center=True).mean().rolling(2,center = True).mean()
plt.figure()
plt.plot(data_high)
plt.plot(rolling_data)


plt.figure()
plt.plot(np.abs(rolling_data-data_cma24))

#c

def mse(x,y):
    diff = [(x[i] - y[i]) for i in range(len(x))]
    diff =  np.array(diff)[~np.isnan(diff)]
    mse = 0
    for i in range(len(diff)):
        mse += diff[i] ** 2
    return mse/len(diff)


def mae(x,y):
    diff = [(x[i] - y[i]) for i in range(len(x))]
    diff =  np.array(diff)[~np.isnan(diff)]
    mae = 0
    for i in range(len(diff)):
        mae += np.abs(diff[i]) 
    return mae/len(diff)


#def sse(x, y):
#    return np.sum(np.power(x - y,2))

MSE1 = mse(data_high,data_cma24)
print("MSE1: ",MSE1)
RMSE1 = cmath.sqrt(MSE1)
print("RMSE1: ",MSE1)

MAE1 = mae(data_high,data_cma24)


MSE2 = mse(data_high,rolling_data)
RMSE2 = cmath.sqrt(MSE2)
MAE2 = mae(data_high,rolling_data)
#d

#y_325 = 0.2 * sum([data_high[i] for i in np.arange(-9,-4)])
#print('Fourth last month: ' , y_325)
#
#
#y_326 = 0.2 * sum([data_high[i] for i in range(-8,-4)]+[y_325])
#print('Third last month: ' , y_326)
#
#
#y_327 = 0.2 * sum([data_high[i] for i in range(-7,-4)]+[y_325,y_326])
#print('Second last month: ' , y_327)
#
#
#y_328 = 0.2 * sum([data_high[i] for i in range(-6,-4)]+[y_325,y_326,y_327])
#print('First last month: ' , y_328)


y_325 = data_high[320:325].mean()
y_326 = data_high[321:326].mean()
y_327 = data_high[322:327].mean()
y_328 = data_high[323:328].mean()
#y_325 = (high[320] + high[321] + high[322] + high[323] + high[324]) / 5
#e
#y_t_4 = data_high[:320]
#y_t_4 = data_high[:-9]
#y_t_3 = data_high[1:-8]
#y_t_2 = data_high[2:-7]
#y_t_1 = data_high[3:-6]
#y_t =  data_high[4:-5]
#y_t_plus_1 = data_high[5:-4]
#
#
#z_t = (y_t_plus_1 - y_t.tshift(1)).as_matrix()
#
#
##z_t_1 = (y_t_1 - y_t.tshift(-1)).as_matrix()
##z_t_2 = (y_t_2 - y_t.tshift(-2)).as_matrix()
##z_t_3 = (y_t_3 - y_t.tshift(-3)).as_matrix()
##z_t_4 = (y_t_4 - y_t.tshift(-4)).as_matrix()
#
#z_t_1 = (y_t_1 - y_t.tshift(-1)).values
#z_t_2 = (y_t_2 - y_t.tshift(-2)).values
#z_t_3 = (y_t_3 - y_t.tshift(-3)).values
#z_t_4 = (y_t_4 - y_t.tshift(-4)).values
#
#X = pd.DataFrame([z_t_1,z_t_2,z_t_3,z_t_4]).transpose()
#
from sklearn.linear_model import LinearRegression
#
#y = np.reshape(z_t, (len(z_t), 1))



y4 = data_high[:320]
y3 = data_high[1:321]
y2 = data_high[2:322]
y1 = data_high[3:323]
y0 =  data_high[4:324]
y_plus_1 = data_high[5:325]

m = y_plus_1.values - y0.values

y = np.reshape(m, (len(m), 1))

z1 = y1.values - y0.values
z2 = y2.values - y0.values
z3 = y3.values - y0.values
z4 = y4.values - y0.values


X = pd.DataFrame([z1,z2,z3,z4]).transpose()



#
lm = LinearRegression(fit_intercept = False)
lm.fit(X, y)

w1,w2,w3,w4= lm.coef_[0]

w0 = 1-(sum(lm.coef_.tolist()[0]))

#y_4_e = w0 * data_high[-5] + w1 * data_high[-6] + w2 * data_high[-7] + w3 * data_high[-8] + w4 * data_high[-9]  
#y_3_e = w0 * y_4_e + w1 * data_high[-5] + w2 * data_high[-6] + w3 * data_high[-7] + w4 * data_high[-8]
#y_2_e = w0 * y_3_e + w1 * y_4_e + w2 * data_high[-5] + w3 * data_high[-6] + w4 * data_high[-7]
#y_1_e = w0 * y_2_e + w1 * y_3_e + w2 * y_4_e + w3 * data_high[-5] + w4 * data_high[-6]

y_4_e = w0 * data_high[324] + w1 * data_high[323] + w2 * data_high[322]+ w3 * data_high[321]+ w4 * data_high[320]  
y_3_e = w0 * data_high[325] + w1 * data_high[324] + w2 * data_high[323]+ w3 * data_high[322]+ w4 * data_high[321]  
y_2_e = w0 * data_high[326] + w1 * data_high[325] + w2 * data_high[324]+ w3 * data_high[323]+ w4 * data_high[322]  
y_1_e = w0 * data_high[327] + w1 * data_high[326] + w2 * data_high[325]+ w3 * data_high[324]+ w4 * data_high[323]


#(f)

# rmse / mae for d

x_est_d = [y_325,y_326,y_327,y_328]
x_test = data_high[-4:]
d_mse = mse(x_est_d,x_test)
d_rmse = cmath.sqrt(d_mse)
d_mae = mae(x_est_d,x_test)

# rmse / mae for e

x_est_e = [y_4_e,y_3_e,y_2_e,y_1_e]
e_mse = mse(x_est_e,x_test)
e_rmse = cmath.sqrt(e_mse)
e_mae = mae(x_est_e,x_test)









