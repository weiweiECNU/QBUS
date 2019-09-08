#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 22:09:28 2019

@author: Viono
"""

#%%==============================================
#task1
#Q1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
wine = pd.read_csv("Wine_2019.csv")
wine.head()
wine.info()
#%%
wine.describe()
#%%
wine.corr()

wine.corr()['quality'].abs().sort_values().round(4)

#fixed acidity and free sulfur dioxide are the most related features with quality
#需要plot选出来的fetures吗？怎么分析？

#%%=============================================
#Q2
#先把三个features相关列提取出来
list = ['density','residual sugar', 'volatile acidity']
wine_list = wine[list]
wine_list.head()

X1 = wine_list.values


y1 = wine['quality'].values

y1 = np.reshape(y1,(-1,1))

#%%
#tut 代码， 但是不知道怎么运用 出不来结果
def Gradient_Descent_Algo(X1, y1, beta, alpha, N, numIterations):
    xTrans = X1.transpose()
    for i in range(0, numIterations):
        # predicted values from the model
        model_0 = np.dot(X1, beta)
        loss_temp = model_0 - y1
        # calculte the loss function
        loss = np.sum(np.square(loss_temp)) / (2 * N)
        # save all the loss function values at each step
        loss_total[i]= loss
        #You can check iteration by
        #print("Iteration: {0} | Loss fucntion: {1}".format(i, loss))
        # calcualte the gradient using matrix representation
        gradient = np.dot(xTrans, loss_temp) / N
        gradient_total[i,:] = gradient.transpose()
        # update the parameters simulteneously with learning rate alpha
        beta = beta - alpha * gradient
        # save all the estimated parametes at each step
        beta_total[i,:]= beta.transpose()
    return beta,loss



#%%
#alternative
from sklearn.metrics import mean_squared_error
def Gradient_Descent_Algo_2(X1, y1, beta, alpha, N, numIterations):
    for i in range(numIterations):
        beta = beta - alpha / N * np.dot(X1.transpose(), (np.dot(X1, beta) - y1))#function
        loss = mean_squared_error(y1, np.dot(X1, beta))/2
        beta_total[i,:] = beta.transpose()#存储循环中T的信息，
        loss_total[i] = loss
    return beta, loss



#%%
numIterations= 20000
alpha = 0.0005
N = len(X1)
loss_total= np.zeros((numIterations,1))#存储，，
beta_total= np.zeros((numIterations,3))
beta_initial = np.reshape(np.zeros(3),(3, 1))

beta = Gradient_Descent_Algo_2(X1, y1, beta_initial, alpha, N, numIterations)[0]
loss= Gradient_Descent_Algo_2(X1, y1, beta_initial, alpha, N, numIterations)[1]

print(beta)
print(loss)
print(beta_total.shape)#modifition algorithm
print(beta_total.shape)


#%%


fig1 = plt.figure()
plt.plot(loss_total, label = "Loss fucntion")
plt.plot(beta_total[:,0], label = "Beta0")
plt.plot(beta_total[:,1], label = "Beta1")
plt.plot(beta_total[:,2], label = "Beta2")
plt.legend(loc="upper right")
plt.xlabel("Number of iteration")
plt.show()

#%%
numIterations= 20000
alpha = 0.0025
N = len(X1)
loss_total= np.zeros((numIterations,1))
beta_total= np.zeros((numIterations,3))
beta_initial = np.reshape(np.zeros(3),(3, 1))

beta = Gradient_Descent_Algo_2(X1, y1, beta_initial, alpha, N, numIterations)[0]
loss= Gradient_Descent_Algo_2(X1, y1, beta_initial, alpha, N, numIterations)[1]

print(beta)
print(loss)
print(beta_total.shape)
print(beta_total.shape)

#%%


fig1 = plt.figure()
plt.plot(loss_total, label = "Loss fucntion")
plt.plot(beta_total[:,0], label = "Beta0")
plt.plot(beta_total[:,1], label = "Beta1")
plt.plot(beta_total[:,2], label = "Beta2")
plt.legend(loc="upper right")
plt.xlabel("Number of iteration")
plt.show()

#怎么解释两个plots
#%%
#不知道怎么解释
numIterations= 20000
alpha = [0.025, 0.01, 0.005, 0.001, 0.0005, 0.0001]
N = len(X1)

loss= np.zeros((6,1))
beta= np.zeros((6,3))
beta_initial = np.reshape(np.zeros(3),(3, 1))

for i in range(6):
    beta[i,:] = Gradient_Descent_Algo_2(X1, y1, beta_initial, alpha[i], N, numIterations)[0].transpose()
    loss[i,:] = Gradient_Descent_Algo_2(X1, y1, beta_initial, alpha[i], N, numIterations)[1]
print(beta[i,:])

#%%==========================================
#Q3
X = wine.iloc[:,:-1]
  
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.20, random_state = 0)

#%%
#with the intercept term
from sklearn.linear_model import LinearRegression
lr_obj_1 = LinearRegression()
lr_obj_1.fit(X_train, y_train)

#%%
#without the intercept term
lr_obj_2 = LinearRegression(fit_intercept = False)
lr_obj_2.fit(X_train, y_train)
    
#%%
y_predict_1 = lr_obj_1.predict(X_test)
y_predict_2 = lr_obj_2.predict(X_test)

print(lr_obj_1.intercept_)
print(lr_obj_1.coef_)

#%%
print(lr_obj_2.intercept_)
print(lr_obj_2.coef_)

#%%
def loss_ols (y1, predict):
    loss = np.dot((y1 - predict).transpose(),(y1 - predict))/(2 * len(y1))
    return loss

print('OLS_1: {0:.4f}'. format(loss_ols(y_test, y_predict_1)[0,0]))
print('OLS_2: {0:.4f}'. format(loss_ols(y_test, y_predict_2)[0,0]))

#%%===========================
#Q4
y_train_ave = np.mean(y_train)
y_train_new = y_train - y_train_ave
y_test_new = y_test - y_train_ave

X_train_ave = np.mean(X_train)
X_train_new = X_train - X_train_ave
X_test_new = X_test - X_train_ave

#%%=======with the intercept term
lr_obj_3 = LinearRegression()
lr_obj_3.fit(X_train_new, y_train_new)

y_predict_3 = lr_obj_3.predict(X_test_new)

print(lr_obj_3.intercept_)
print(lr_obj_3.coef_)

#%%=======without the intercept term
lr_obj_4 = LinearRegression(fit_intercept = False)
lr_obj_4.fit(X_train_new, y_train_new)

y_predict_4 = lr_obj_4.predict(X_test_new)

print(lr_obj_4.intercept_)
print(lr_obj_4.coef_)

#%%

print('OLS_3: {0:.4f}'. format(loss_ols(y_test_new, y_predict_3)[0,0]))
print('OLS_4: {0:.4f}'. format(loss_ols(y_test_new, y_predict_4)[0,0]))


#%%=================================
#Q5
high_quality_random = wine[wine.iloc[:,-1]>6].sample(n=400, random_state = 0)
low_quality_random = wine[wine.iloc[:,-1]<=6].sample(n=400, random_state = 0)

all_random  = pd.concat([high_quality_random, low_quality_random])

#%%
X2 = all_random.iloc[:,:-1]
y2 = all_random.iloc[:,-1]
X2_train_lasso, X2_test_lasso, y2_train_lasso, y2_test_lasso = train_test_split(X2, y2, test_size = 0.20, random_state = 0)

#%%
X2_train_ave_lasso = np.mean(X2_train_lasso)
X2_train_new_lasso = X2_train_lasso - X2_train_ave_lasso
X2_test_new_lasso = X2_test_lasso - X2_train_ave_lasso

y2_train_ave_lasso = np.mean(y2_train_lasso)
y2_train_new_lasso = y2_train_lasso - y2_train_ave_lasso
y2_test_new_lasso = y2_test_lasso - y2_train_ave_lasso

#%%
from sklearn.linear_model import LassoCV
lasso_1 = LassoCV()
lasso_1.fit(X2_train_new_lasso, y2_train_new_lasso)

predict_lasso1 = lasso_1.predict(X2_test_new_lasso)

print(lasso_1.intercept_)
print(lasso_1.coef_)

print('Lasso Lambda: {0:.4f}'. format(lasso_1.alpha_))

#%%
print('Lasso: {0:.4f}'. format(mean_squared_error(y2_test_new_lasso, predict_lasso1)/2))

#%%
lasso_2 = LassoCV(alphas = [lasso_1.alpha_])
lasso_2.fit(X2_train_new_lasso, y2_train_new_lasso)

predict_lasso2 = lasso_2.predict(X2_test_new_lasso)

print(lasso_2.intercept_)
print(lasso_2.coef_)

print('Lasso Lambda: {0:.4f}'. format(lasso_2.alpha_))

#%%
print('Lasso: {0:.4f}'. format(mean_squared_error(y2_test_new_lasso, predict_lasso2)/2))


#%%========================================
#taskB
#Q1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#%%
loans = pd.read_excel('Loans_Data_2019.xlsx')

#%%
loans['debt_settlement_flag'] = np.where(loans.iloc[:,-1] == 'Y',1,0)


#%%
#???
loans.info()
loans.describe()

if loans.isnull().values.any():
    print('Contains missing values')
else:
    print('No missing values')
    
#%%
loans.skew()

#%%
loans.kurtosis()

#%%
plt.figure()
plt.hist(loans.iloc[:, 0])
plt.show()

#%%
plt.figure()
plt.hist(loans.loc[:, 'total_rec_int'])
plt.show()

#%%
loans.corr()
#%%

plt.figure()
plt.scatter(loans.loc[:, 'total_rec_int'], loans.iloc[:,-1]);
plt.show()

#%%========================================
#taskB
#Q2
loans_1 = loans[['loan_amnt', 'annual_inc', 'int_rate', 'installment']]

X = loans_1.values

y = loans.iloc[:,-1].values
y = y.reshape(len(y),1)

#%%
X_ave = X.mean().reshape(-1,1)
X_stdv = X.std().reshape(-1,1)
X1 = np.divide(np.subtract(X, X_ave), X_stdv)

#%%
from sklearn.metrics import mean_squared_error

def Gradient_Descent_Algo(X, y, beta, alpha, N, numIterations):
    xTrans = X.transpose()
    for i in range(0, numIterations):
        # predicted values from the model
        model_0 = np.dot(X, beta)
        loss_temp = model_0 - y
        # calculte the loss function
        loss = np.sum(np.square(loss_temp)) / (2 * N)
        # save all the loss function values at each step
        loss_total[i]= loss
        #You can check iteration by
        #print("Iteration: {0} | Loss fucntion: {1}".format(i, loss))
        # calcualte the gradient using matrix representation
        gradient = np.dot(xTrans, loss_temp) / N
        gradient_total[i,:] = gradient.transpose()
        # update the parameters simulteneously with learning rate alpha
        beta = beta - alpha * gradient
        # save all the estimated parametes at each step
        beta_total[i,:]= beta.transpose()
    return beta,loss

#%%
def sigmoid(x):
    return(1/(1+ np.exp(-x)))
    
def logistic_loss(x, beta, y_test):
    left = np.log(sigmoid(np.dot(x, beta))) * y_test.reshape(-1,1)
    right = np.log(1 - sigmoid(np.dot(x, beta))) * (1 - y_test.reshape(-1,1))
    loss = -np.mean(left + right)
    return loss

def myLogisticGD(X, t, beta, alpha, N, T):
    xTrans = X.transpose()
    for i in range(T):
        beta = beta - alpha / N * np.dot (xTrans, (sigmoid(np.dot(X, beta)) - y))
        loss = logistic_loss (X, beta, y)
        beta_total[i, :] = beta.transpose()
        loss_total[i] = loss
    return beta, loss

#%%
T = 10000
alpha = 0.0005
N = len(X1)
beta_initial = np.reshape(np.zeros(4),(4,1))

beta_total = np.zeros((T, 4))
loss_total = np.zeros(T)

beta = myLogisticGD(X1, y, beta_initial, alpha, N, T)[0]
loss = myLogisticGD(X1, y, beta_initial, alpha, N, T)[1]

#%%
print(loss)
print(beta)
print(loss_total.shape)
print(beta_total.shape)

#%%
fig1 = plt.figure()
plt.plot(loss_total, label = "Loss fucntion")
plt.plot(beta_total[:,0], label = "Beta0")
plt.plot(beta_total[:,1], label = "Beta1")
plt.plot(beta_total[:,2], label = "Beta2")
plt.plot(beta_total[:,3], label = "Beta3")
plt.legend(loc="upper right")
plt.xlabel("Number of iteration")
plt.show()


#%%=====================================
#Q3
X2 = loans[['int_rate', 'funded_amnt', 'total_rec_prncp']].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size = 0.20, random_state = 0)

#%%
from sklearn.linear_model import LogisticRegression

lg_res = LogisticRegression()
lg_res.fit(X_train, y_train)

lg_res.coef_

#%%
prob = lg_res.predict_proba(X_test)
prob_1 = prob[:, 0]

#%%
def sigmoid(x):
    return (1 / (1+np.exp(-x)))

def logistic_loss(x, beta, y_test):
    left = np.log(sigmoid(np.dot(x,beta))) * y_test.reshape(-1,1)
    right = np.log(1 - sigmoid(np.dot(x, beta))) * (1 - y_test.reshape(-1,1))
    loss = -np.mean(left + right)
    return loss

#%%
loss_lg = logistic_loss(X_test, lg_res.coef_.transpose(), y_test).round(4)
loss_lg