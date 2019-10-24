#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 03:24:02 2019

@author: Viono
"""


# %%========
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.linear_model import LassoCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from scipy import stats
from scipy.stats import norm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time


warnings.filterwarnings('ignore')
comments = pd.read_csv('User_Comments.csv')
#print(comments.shape)
#comments.head()
# %%
features = comments.loc[:, 'CONTENT']
target = comments.loc[:, 'CLASS']

# %%
np.average(target)

# %%
tfidf_vectorizer = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2), min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(features)
# %%
#np.shape(tfidf_vectorizer.get_feature_names())
#
#tfidf_vectorizer.vocabulary_

print(tfidf_vectorizer.get_feature_names())

print(tfidf.shape)



# %%
X_train, X_test, y_train, y_test = train_test_split(
    tfidf, target, test_size=0.2, random_state=0)

# %%=====(b)====

# %%
parameters = {'n_estimators': np.arange(1, 200, 5)}

RFCV = GridSearchCV(ensemble.RandomForestClassifier(
    class_weight='balanced', random_state=0), parameters, cv=5, return_train_score=True)

tic = time.time()
RFCV.fit(X_train, y_train)
toc = time.time()
print("Training time: {0:.4f}s".format(toc - tic))

best_es = RFCV.best_estimator_
print(best_es)


# %%
re_fit = ensemble.RandomForestClassifier(
    class_weight='balanced', random_state=0, n_estimators=141)
re_fit.fit(X_train, y_train)

# %%
y_pred = re_fit.predict(X_test)


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# %%=====(c)======
tree_number = np.arange(1, 200, 5)
mean_train_score = RFCV.cv_results_['mean_train_score']
mean_test_score = RFCV.cv_results_['mean_test_score']
#mean_test_score.shape

# %%
plt.figure()
plt.plot(tree_number, mean_train_score, "g", label='mean_train_score')
plt.plot(tree_number, mean_test_score, "r", label='mean_test_score')
plt.xlabel('Number of Trees')
plt.legend()
plt.title('Mean of Train and Test Dataset')
plt.show()


# Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.

# %%
print(RFCV.best_params_)
RFCV.best_score_.round(4)

# %%=====(d)=====
depth = [decisionTree.tree_.max_depth for decisionTree in re_fit.estimators_]

# %%
plt.figure()
plt.hist(depth)
plt.title("Histogram of the depths of the trees")
plt.xlabel("Depth")
plt.ylabel("Number")
plt.show()

## 

# %%=====(e)=======
top_idx = np.argsort(re_fit.feature_importances_)[::-1][:10]

features = [tfidf_vectorizer.get_feature_names()[i] for i in top_idx]
importances = [round(re_fit.feature_importances_[i],4) for i in top_idx]

pd.DataFrame({"Importance" : importances}, index=features )



# %%
# 画不出来
plt.title('Top10 Important Features')
plt.barh(range(10), importances, color='b', align='center')
plt.yticks(range(10), features)  # removed[indices]
plt.xlabel("Relative Importance")
plt.show()


# %%======taskB

# %%
train = pd.read_csv('NBA_Train.csv')
test = pd.read_csv('NBA_Test.csv')


# %%
train.info()
test.info()
test.describe()


col_category = ["POSITION", "TEAM"]
col_numerical = list(train.columns.drop(["Train_ID","TEAM", "POSITION","SALARY"]))


# %%
############################
# preprogressing
############################
if train.isnull().values.any():
    print('Contains missing values')
else:
    print('No missing values')
    
# outiliers
    
#plt.figure(figsize=(30, 90))
#num_subplot = len(col_numerical+["SALARY"])
#
#for i, col in enumerate(col_numerical+["SALARY"]):
#    plt.subplot(num_subplot//2 + 1, 2, i+1)
#    sns.set_palette("pastel")
#    sns.boxplot(x=train[col])
#    
#plt.savefig("Boxplot of numeriacal variables and target variables")



plt.figure(figsize=(30, 90))
num_subplot = len(col_numerical+["SALARY"])

for i, col in enumerate(col_numerical+["SALARY"]):
    plt.subplot(num_subplot//2 + 1, 2, i+1)
    plt.hist(train[col])
    plt.xlabel(col)
    plt.ylabel("Number")
    
plt.savefig("Histgram of numeriacal variables and target variables")





# %%
    
############################
# eda
############################ 

def get_status(x):
    return pd.DataFrame([x.count(), x.mean(),  x.min(), x.quantile(.25), x.quantile(.5), x.quantile(.75), x.max(), x.median(), x.mad(), x.var(), x.std(), x.skew(), x.kurt(), ],
                        index=['count', 'mean', 'min', '25%', '50%', '75%', 'max', 'median', 'mad', 'var', 'std', 'skew', 'kurt', ]).round(3)
    
def plot_distributed(series):
    
    sns.distplot(series, fit=norm, bins=10, color='cornflowerblue')
    (mu, sigma) = norm.fit(series)
    plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=${:.2f})'.format(
        mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Distribution of '+series.name)
    
## stats values
    
print(get_status(train[col_numerical+["SALARY"]]))

### box-plot
plt.figure(figsize=(30, 90))
num_subplot = len(col_numerical+["SALARY"])

for i, col in enumerate(col_numerical+["SALARY"]):
    plt.subplot(num_subplot//2 + 1, 2, i+1)
    sns.set_palette("pastel")
    sns.boxplot(x=train[col])
    
plt.savefig("Boxplot of numeriacal variables and target variables")


# distribution plot
plt.figure(figsize=(30, 90))
num_subplot = len(col_numerical+["SALARY"])
for i, col in enumerate(col_numerical+["SALARY"]):
    plt.subplot(num_subplot//2 + 1, 2, i+1)
    sns.set_palette("pastel")
    
    plot_distributed(train[col])
plt.savefig("Distribution of numeriacal variables and target variables")

# prob

plt.figure(figsize=(30, 90))
num_subplot = len(col_numerical+["SALARY"])
for i, col in enumerate(col_numerical+["SALARY"]):
    plt.subplot(num_subplot//2 + 1, 2, i+1)
    sns.set_palette("pastel")
    stats.probplot(train[col],plot=plt)
    
plt.savefig("Q-Q plot of numeriacal variables and target variables")


# %%


## SALARY 

print(get_status(train["SALARY"]**(1/3)))

### box-plot
plt.figure()
sns.boxplot(x=train["SALARY"]**(1/3))
plt.savefig("Boxplot of salary 3")


# distribution plot
plt.figure()
plot_distributed(train["SALARY"]**(1/3))
plt.savefig("Distribution plot of salary 3")

# prob

plt.figure()
stats.probplot(train["SALARY"]**(1/3),plot=plt)
plt.savefig("Q-Q plot of salary 3")

# %%

#col_numerical

df_eda = train[col_numerical]

skew_high_cols = ["BLK", "OWS", "WS"]
skew_medium_cols = ["PER", "ORB", "DRB", "TRB", "AST", "STL", "TOV", "DWS"]
skew_low_log = ["Games"]

for col in skew_high_cols:
    df_eda[col] = np.log(df_eda[col]+2)

for col in skew_medium_cols:
    df_eda[col] = np.sqrt(df_eda[col])

skew_low_log = ["Games"]
for col in skew_low_log:
    df_eda[col] = df_eda[col] ** 2



## stats values
    
print(get_status(df_eda))

### box-plot
plt.figure(figsize=(30, 500))
num_subplot = len(df_eda)

for i, col in enumerate(df_eda):
    plt.subplot(num_subplot//2 + 1, 2, i+1)
    sns.set_palette("pastel")
    sns.boxplot(x=df_eda[col])
    
plt.savefig("Boxplot of numeriacal variables transformed")


# distribution plot
plt.figure(figsize=(30, 500))
num_subplot = len(df_eda)
for i, col in enumerate(df_eda):
    plt.subplot(num_subplot//2 + 1, 2, i+1)
    sns.set_palette("pastel")
    
    plot_distributed(df_eda[col])
plt.savefig("Distribution of numeriacal variables transformed")

# prob

plt.figure(figsize=(30, 500))
num_subplot = len(df_eda)
for i, col in enumerate(df_eda):
    plt.subplot(num_subplot//2 + 1, 2, i+1)
    sns.set_palette("pastel")
    stats.probplot(df_eda[col],plot=plt)
    
plt.savefig("Q-Q plot of numeriacal variables transformed")

# %%


# corr

df_eda = pd.concat([df_eda,np.log(train["SALARY"])],axis=1)
plt.figure(figsize = (10,8))
sns.heatmap(df_eda.corr(), vmin = 0, vmax = 1, cmap = 'Reds')
plt.title('Correlation Among Variables', fontsize = 16)
plt.savefig("Correlation Among numerical Variables")

print(abs(df_eda.corr()["SALARY"]).sort_values(ascending=False))


# %%

plt.figure(figsize=(16,8))
plt.scatter(train["Games"], train["Minutes"])
plt.xlabel("Games")
plt.ylabel("Minutes")

items = ["Games", " vs ", "Minutes"]
title = ''.join(items)
plt.title(title)
plt.savefig(title)



plt.figure(figsize=(16,8))
plt.scatter(train["TRB"], train["ORB"])
plt.xlabel("TRB")
plt.ylabel("ORB")

items = ["TRB", " vs ", "ORB"]
title = ''.join(items)
plt.title(title)
plt.savefig(title)


plt.figure(figsize=(16,8))
plt.scatter(train["TRB"], train["DRB"])
plt.xlabel("TRB")
plt.ylabel("DRB")

items = ["TRB", " vs ", "DRB"]
title = ''.join(items)
plt.title(title)
plt.savefig(title)




plt.figure(figsize=(16,8))
plt.scatter(train["WS"], train["DWS"])
plt.xlabel("WS")
plt.ylabel("DWS")

items = ["WS", " vs ", "DWS"]
title = ''.join(items)
plt.title(title)
plt.savefig(title)


plt.figure(figsize=(16,8))
plt.scatter(train["WS"], train["OWS"])
plt.xlabel("WS")
plt.ylabel("OWS")

items = ["WS", " vs ", "OWS"]
title = ''.join(items)
plt.title(title)
plt.savefig(title)

# %%

plt.figure(figsize=(20,10))
sns.set_palette("pastel")
sns.boxplot(x = 'TEAM', y = 'SALARY', data = train)
sns.despine()
plt.title('Correlation between SALARY and TEAM', fontsize = 16)
plt.xticks(rotation=270)
plt.savefig("Boxplot of Team")


sns.set_palette("pastel")
sns.boxplot(x = 'POSITION', y = 'SALARY', data = train)
sns.despine()
plt.title('Correlation between SALARY and POSITION', fontsize = 16)
plt.savefig("Heatmap of Nymerical Variables vs Target")




# %%

temp = pd.DataFrame()
for col in col_category:
    dummies = pd.get_dummies(
        train[col], prefix_sep="_", drop_first=True, prefix=col)
    #df.drop(columns=col, inplace=True)
    temp = pd.concat([temp, dummies], axis=1)


temp = pd.concat([temp,train["SALARY"]], axis=1)

plt.figure(figsize=(10, 8))
sns.heatmap(temp.corr(), vmin=0, vmax=1, cmap='Reds')
plt.title('Correlation Among Category Variables', fontsize=16)
plt.savefig("Heatmap of Category Variables vs Target")

print(abs(temp.corr()["SALARY"]).sort_values(ascending=False))
# %%
############################
# feature eng
############################    
    

skew_high_cols = ["BLK", "OWS", "WS"]
skew_medium_cols = ["PER", "ORB", "DRB", "TRB", "AST", "STL", "TOV", "DWS"]
skew_low_log = ["Games"]



def fe_target_var(df):
#     return pd.DataFrame(np.log(df["SALARY"]))
#     return pd.DataFrame(df["SALARY"])
    return pd.DataFrame(df["SALARY"]**(1/3))

def fe_numerical_var(df):

# 比较强的降低偏度
    df_number = df[col_numerical+["SALARY"]]
    for col in skew_high_cols:
        df_number[col] = np.log(df_number[col]+2)
# 稍微降低偏度    
    for col in skew_medium_cols:
        df_number[col] = np.sqrt(df_number[col])
#左移 增大偏度
    for col in skew_low_log:
        df_number[col] = df_number[col] ** 2

    df_number.drop(columns=["TRB",
                            "WS"], axis=1, inplace=True)

    return df_number




def fe_categorical_var(df, cols):
    d = pd.DataFrame()
    for col in cols:
        dummies = pd.get_dummies(
            df[col], prefix_sep="_", drop_first=True, prefix=col)
        d = pd.concat([d, dummies], axis=1)

    return d


def feature_eng(df):
    return pd.concat(
        [fe_target_var(df), fe_numerical_var(df), fe_categorical_var(df, col_category)], axis=1)




train_dumy = feature_eng(pd.read_csv('NBA_Train.csv'))
test_dumy = feature_eng(pd.read_csv('NBA_Test.csv'))


# %%
list(set(train_dumy.columns.values)-set(test_dumy.columns.values))

# %%

train_dumy = train_dumy.drop('TEAM_Houston Rockets', axis=1)



# %%
train_X = train_dumy.iloc[:, 1:]
train_t = train_dumy.iloc[:, 0]

test_X = test_dumy.iloc[:, 1:]
test_t = test_dumy.iloc[:, 0]


# %%


# %%
las = LassoCV()
las.fit(train_X, train_t)

# %%
predict_t = las.predict(test_X)


# %%
print(las.intercept_)
print(las.coef_)
# ??跟顶点答案不一样


# %%
print('LASSO Lambda: {0:.4f}'.format(las.alpha_))

# check the prediction accuracy

# %%
def RMSE(a, b):
    return np.sqrt(np.mean(np.power(a-b, 2)))


RMSE(predict_t ** 3, test_t ** 3)


# %%
# method2=======xgboost

# %%
xgb_model = xgb.XGBRegressor()
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
gridsch = GridSearchCV(xgb_model,
                       {'max_depth': [2, 4, 6],
                        'n_estimators': [50, 100, 200],
                        'learning_rate': learning_rate}, verbose=1)
result = gridsch.fit(train_X, train_t)
# %%
print(result.best_score_)
print(result.best_params_)

# %%
xgb_para = result.best_estimator_
xgb_para = xgb_para.fit(train_X, train_t)
xgb_para

# %%
xgb_para.feature_importances_
# 跟顶点答案有差

predict_xgb = xgb_para.predict(test_X)

np.sqrt(mean_squared_error(predict_xgb ** 3, test_t ** 3))

#为什么这个用的是test-t, 上面那个方法用的是new
