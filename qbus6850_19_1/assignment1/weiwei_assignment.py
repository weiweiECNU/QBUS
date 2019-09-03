# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_wine = pd.read_csv("Wine_New.csv")

#Task1

#1

#(1)
print(" The number of poor-quality wine: ",len(data_wine[ data_wine['quality'] < 6 ]))


#(2)
wine_features_df = data_wine[['quality','density','residual sugar','volatile acidity']]
wine_features_df.corr()

#2(1)
from sklearn.model_selection import train_test_split

X = data_wine.iloc[:,:-1]
y = data_wine.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

#(2)
from sklearn.linear_model import LinearRegression

lr_obj_intercept = LinearRegression(fit_intercept = True)
lr_obj_intercept.fit(X_train, y_train)

lr_obj_no_intercept = LinearRegression(fit_intercept = False)
lr_obj_no_intercept.fit(X_train, y_train)

#print(lr_obj_intercept.intercept_)
## For other beta
#print(lr_obj_intercept.coef_)        
#
#X_test_add_one = np.column_stack((X_test,np.ones(len(X_test))))
#coef_add_interceft = np.append(lr_obj_intercept.coef_,lr_obj_intercept.intercept_)
#
#X_test_add_one = np.column_stack((X_test,np.ones(len(X_test))))
#
#model_0 = np.dot(X_test_add_one, coef_add_interceft)
#
#loss_temp = model_0 - y_test
#
#loss = np.sum(np.square(loss_temp)) / (2 * (len(X_test)))


def get_loss(x_test, y_test, intercept, coef):
    
    X_test_add_one = np.column_stack((x_test,np.ones(len(x_test))))
    
    coef_add_interceft = np.append(coef,intercept)

    model_0 = np.dot(X_test_add_one, coef_add_interceft)

    loss_temp = model_0 - y_test

    return np.sum(np.square(loss_temp)) / (2 * (len(X_test)))

print("The loss of regression with intercept: ",get_loss(X_test,y_test,lr_obj_intercept.intercept_,lr_obj_intercept.coef_))
print("The loss of regression without intercept: ",get_loss(X_test,y_test,lr_obj_no_intercept.intercept_,lr_obj_no_intercept.coef_))

#3
centred = lambda x: x-x.mean()

centred_y_train = centred(y_train)

new_FA_train = centred(X_train['fixed acidity'])

new_VA_train = centred(X_train['volatile acidity'])

new_CA_train = centred(X_train['critric acid'])

new_RS_train = centred(X_train['residual sugar'])

new_CH_train = centred(X_train['chlorides'])

new_FSD_train = centred(X_train['free sulfur dioxide'])

new_TSD_train = centred(X_train['total sulfur dioxide'])

new_DS_train = centred(X_train['density'])

new_PH_train = centred(X_train['pH'])

new_SP_train = centred(X_train['sulphates'])

new_AL_train = centred(X_train['alcohol'])

centred_X_train = pd.DataFrame([new_FA_train,new_VA_train,new_CA_train,new_RS_train,new_CH_train,new_FSD_train,new_TSD_train,new_DS_train,new_PH_train,new_SP_train,new_AL_train]).transpose()

lr_obj_centred = LinearRegression()
lr_obj_centred.fit(centred_X_train, centred_y_train)

print("The coefficients of regression with centred data: ", lr_obj_centred.coef_)
print("The intercept of regression with centred data: ", lr_obj_centred.intercept_)

centred_X_test = centred( X_test)
centred_y_test = centred( y_test)

print("The loss of regression with centred data: ",get_loss(centred_X_test,centred_y_test,lr_obj_centred.intercept_, lr_obj_centred.coef_))

#4(1)

poor_quality_wine  = data_wine[ data_wine['quality'] < 6 ].sample(n=200, random_state=0)

good_quality_wine  = data_wine[ data_wine['quality'] >= 6 ].sample(n=200, random_state=0)

wine_concat = pd.concat([poor_quality_wine,good_quality_wine])

#（2）
from sklearn.linear_model import LassoCV

X_wine = wine_concat.iloc[:,:-1]
y_wine = wine_concat.iloc[:,-1]

X_wine_centred = centred( X_wine )
y_wine_centred = centred( y_wine )

X_lasso_train, X_lasso_test, y_lasso_train, y_lasso_test = train_test_split(X_wine_centred, y_wine_centred,test_size=0.2,random_state=0)


lascv = LassoCV(cv=5, random_state=0)
lascv.fit(X_lasso_train, y_lasso_train)

print("LASSO Lambda: {0}".format(lascv.alpha_))

def get_loss_lasso(x_test, y_test, intercept, coef, Lambda):
    
    X_test_add_one = np.column_stack((x_test,np.ones(len(x_test))))
    
    coef_add_interceft = np.append(coef,intercept)

    model_0 = np.dot(X_test_add_one, coef_add_interceft)

    loss_temp = model_0 - y_test

    lasso = Lambda * ( np.sum( np.abs(coef_add_interceft) ) )
    return (np.sum(np.square(loss_temp)) + lasso )/ (2 * (len(X_test))) 

preds_lasso = lascv.predict(X_lasso_test)

print("The loss of regression with lasso: ",get_loss_lasso(X_lasso_test,y_lasso_test,lascv.intercept_, lascv.coef_,lascv.alpha_ ))


