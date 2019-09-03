# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings
from statlearning import plot_feature_importance
import matplotlib.pyplot as plt



df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_test['price'] = 0
df_all = pd.concat([df_train,df_test])
df_all.index = np.arange(2000)