# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_clean = train.iloc[:,2:]
test_clean = test.iloc[:,1:]


train_clean['experiences_offered'] = np.where(train_clean['experiences_offered'].str.contains('none'), "none","experiences")
test_clean['experiences_offered'] = np.where(test_clean['experiences_offered'].str.contains('none'), "none","experiences" )

temp_df = pd.get_dummies(train_clean["experiences_offered"],prefix = "experiences_offered").iloc[:,0]
train_clean = pd.concat([train_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
train_clean.drop(['experiences_offered'], axis=1,inplace = True)

temp_df = pd.get_dummies(test_clean["experiences_offered"],prefix = "experiences_offered").iloc[:,0]
test_clean = pd.concat([test_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
test_clean.drop(['experiences_offered'], axis=1,inplace = True)




train_clean["host_response_time"] = train_clean["host_response_time"].fillna("None")
temp_df = pd.get_dummies(train_clean["host_response_time"],prefix = "host_response_time").iloc[:,1:]
train_clean = pd.concat([train_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
train_clean.drop(['host_response_time'], axis=1,inplace = True)

test_clean["host_response_time"] = test_clean["host_response_time"].fillna("None")
temp_df = pd.get_dummies(test_clean["host_response_time"],prefix = "host_response_time").iloc[:,1:]
test_clean = pd.concat([test_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
test_clean.drop(['host_response_time'], axis=1,inplace = True)





train_clean["host_response_rate"].fillna(0,inplace = True)
test_clean["host_response_rate"].fillna(0,inplace = True)

test_clean["host_is_superhost"].fillna('f',inplace = True)
test_clean["host_listings_count"].fillna(1,inplace = True)
test_clean["host_identity_verified"].fillna('f',inplace = True)

train_clean["host_is_superhost"] = train_clean.host_is_superhost.map({'f':0, 't':1})
test_clean["host_is_superhost"] = test_clean.host_is_superhost.map({'f':0, 't':1})

train_clean["host_identity_verified"] = train_clean.host_identity_verified.map({'f':0, 't':1})
test_clean["host_identity_verified"] = test_clean.host_identity_verified.map({'f':0, 't':1})






train_clean['property_type'] = np.where(train_clean['property_type'].str.contains('House'), "House", (np.where(train_clean['property_type'].str.contains('Apartment'), "Apartment", (np.where(train_clean['property_type'].str.contains('Townhouse'), "Townhouse", "Other")))))
test_clean['property_type'] = np.where(test_clean['property_type'].str.contains('House'), "House", (np.where(test_clean['property_type'].str.contains('Apartment'), "Apartment", (np.where(test_clean['property_type'].str.contains('Townhouse'), "Townhouse", "Other")))))

temp_df = pd.get_dummies(train_clean["property_type"],prefix = "property_type").iloc[:,[0,1,3]]
train_clean = pd.concat([train_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
train_clean.drop(['property_type'], axis=1,inplace = True)

temp_df = pd.get_dummies(test_clean["property_type"],prefix = "property_type").iloc[:,[0,1,3]]
test_clean = pd.concat([test_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
test_clean.drop(['property_type'], axis=1,inplace = True)






temp_df = pd.get_dummies(train_clean["room_type"],prefix = "room_type").iloc[:,:2]
train_clean = pd.concat([train_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
train_clean.drop(['room_type'], axis=1,inplace = True)

temp_df = pd.get_dummies(test_clean["room_type"],prefix = "room_type").iloc[:,:2]
test_clean = pd.concat([test_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
test_clean.drop(['room_type'], axis=1,inplace = True)




train_clean["bathrooms"].fillna(0, inplace=True)
test_clean["bathrooms"].fillna(0, inplace=True) 

train_clean["bedrooms"].fillna(0, inplace=True)
test_clean["bedrooms"].fillna(0, inplace=True) 

train_clean["beds"].fillna(0, inplace=True)
test_clean["beds"].fillna(0, inplace=True) 





train_clean['bed_type'] = np.where(train_clean['bed_type'].str.contains('Real Bed'), "Real Bed","Not Bed")
test_clean['bed_type'] = np.where(test_clean['bed_type'].str.contains('Real Bed'), "Real Bed","Not Bed" )

temp_df = pd.get_dummies(train_clean["bed_type"],prefix = "bed_type").iloc[:,1]
train_clean = pd.concat([train_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
train_clean.drop(['bed_type'], axis=1,inplace = True)

temp_df = pd.get_dummies(test_clean["bed_type"],prefix = "bed_type").iloc[:,1]
test_clean = pd.concat([test_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
test_clean.drop(['bed_type'], axis=1,inplace = True)





train_clean["security_deposit"].fillna(0, inplace=True)
test_clean["security_deposit"].fillna(0, inplace=True)

train_clean["cleaning_fee"].fillna(0, inplace=True)
test_clean["cleaning_fee"].fillna(0, inplace=True)




 
train_clean["review_scores_rating"].fillna(np.mean(train_clean["review_scores_rating"]), inplace=True)
test_clean["review_scores_rating"].fillna(np.mean(test_clean["review_scores_rating"]), inplace=True)

train_clean["review_scores_accuracy"].fillna(np.mean(train_clean["review_scores_accuracy"]), inplace=True)
test_clean["review_scores_accuracy"].fillna(np.mean(test_clean["review_scores_accuracy"]), inplace=True)

train_clean["review_scores_cleanliness"].fillna(np.mean(train_clean["review_scores_cleanliness"]), inplace=True)
test_clean["review_scores_cleanliness"].fillna(np.mean(test_clean["review_scores_cleanliness"]), inplace=True)

train_clean["review_scores_checkin"].fillna(np.mean(train_clean["review_scores_checkin"]), inplace=True)
test_clean["review_scores_checkin"].fillna(np.mean(test_clean["review_scores_checkin"]), inplace=True)

train_clean["review_scores_communication"].fillna(np.mean(train_clean["review_scores_communication"]), inplace=True)
test_clean["review_scores_communication"].fillna(np.mean(test_clean["review_scores_communication"]), inplace=True)

train_clean["review_scores_location"].fillna(np.mean(train_clean["review_scores_location"]), inplace=True)
test_clean["review_scores_location"].fillna(np.mean(test_clean["review_scores_location"]), inplace=True)

train_clean["review_scores_value"].fillna(np.mean(train_clean["review_scores_value"]), inplace=True)
test_clean["review_scores_value"].fillna(np.mean(test_clean["review_scores_value"]), inplace=True)



train_clean["instant_bookable"] = train_clean.instant_bookable.map({'f':0, 't':1})
test_clean["instant_bookable"] = test_clean.instant_bookable.map({'f':0, 't':1})




train_clean['cancellation_policy'] = np.where(train_clean['cancellation_policy'].str.contains('flexible'), "flexible", (np.where(train_clean['cancellation_policy'].str.contains('moderate'), "moderate", (np.where(train_clean['cancellation_policy'].str.contains('strict_14_with_grace_period'), "strict_14_with_grace_period", "super_strict")))))
test_clean['cancellation_policy'] = np.where(test_clean['cancellation_policy'].str.contains('flexible'), "flexible", (np.where(test_clean['cancellation_policy'].str.contains('moderate'), "moderate", (np.where(test_clean['cancellation_policy'].str.contains('strict_14_with_grace_period'), "strict_14_with_grace_period", "super_strict")))))
temp_df = pd.get_dummies(train_clean["cancellation_policy"],prefix = "cancellation_policy").iloc[:,:3]
train_clean = pd.concat([train_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
train_clean.drop(['cancellation_policy'], axis=1,inplace = True)

temp_df = pd.get_dummies(test_clean["cancellation_policy"],prefix = "cancellation_policy").iloc[:,:3]
test_clean = pd.concat([test_clean.reset_index(drop=True), temp_df.reset_index(drop=True)],axis =1)
test_clean.drop(['cancellation_policy'], axis=1,inplace = True)




train_clean["require_guest_profile_picture"] = train_clean.require_guest_profile_picture.map({'f':0, 't':1})
test_clean["require_guest_profile_picture"] = test_clean.require_guest_profile_picture.map({'f':0, 't':1})

train_clean["require_guest_phone_verification"] = train_clean.require_guest_phone_verification.map({'f':0, 't':1})
test_clean["require_guest_phone_verification"] = test_clean.require_guest_phone_verification.map({'f':0, 't':1})




train_clean["reviews_per_month"].fillna(0,inplace = True)
test_clean["reviews_per_month"].fillna(0,inplace = True)


###############################################################################
#
#EDA
#
###############################################################################

train_clean = pd.concat([train_clean.reset_index(drop=True), train["price"].reset_index(drop=True)],axis =1)
corr = train_clean.corr()


cols = abs(corr).nlargest(11, 'price')['price'].index
corr_most_10 = cols[1:].tolist()







############
#
#Model
#
######
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import xgboost as xgb
import lightgbm as lgb
from mlxtend.regressor import StackingCVRegressor

train_clean = pd.concat([train_clean.reset_index(drop=True), train["price"].reset_index(drop=True)],axis =1)

train_eng = train_clean.copy()
train_eng["accommodates"] = np.log(train_clean["accommodates"])
train_eng["beds"] = np.sqrt(train_clean["beds"])
train_eng["bedrooms"] = np.log1p(train_clean["bedrooms"])
train_eng["cleaning_fee"] = np.sqrt(train_clean["cleaning_fee"])
train_eng["bathrooms"] = np.sqrt(train_clean["bathrooms"])
train_eng["guests_included"] = np.log1p(train_clean["guests_included"])
train_eng["price"] = np.log(train_clean["price"])

test_eng = test_clean.copy()
test_eng["accommodates"] = np.log(test_clean["accommodates"])
test_eng["beds"] = np.sqrt(test_clean["beds"])
test_eng["bedrooms"] = np.log1p(test_clean["bedrooms"])
test_eng["cleaning_fee"] = np.sqrt(test_clean["cleaning_fee"])
test_eng["bathrooms"] = np.sqrt(test_clean["bathrooms"])
test_eng["guests_included"] = np.log1p(test_clean["guests_included"])



response='price'
predictors=list(train_eng.columns.values[:-1])
index_train, index_test  = train_test_split(np.array(train_eng.index), train_size=0.8, random_state=0)

train = train_eng.loc[index_train,:].copy()
test =  train_eng.loc[index_test,:].copy()

y_train = train[response]
y_test = test[response]

X_train = train[predictors].copy()
X_test = test[predictors].copy()

X_train_10 = train[corr_most_10].copy()
X_test_10 = test[corr_most_10].copy()

