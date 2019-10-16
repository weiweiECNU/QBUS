#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
import time
import seaborn as sns

from statlearning import plot_regressions
from statlearning import plot_histogram

from scipy import stats
from scipy.stats import norm, skew
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


tic = time.time()

sns.set_context('notebook')
sns.set_style('ticks')
crayon = ['#4E79A7', '#F28E2C', '#E15759', '#76B7B2', '#59A14F',
          '#EDC949', '#AF7AA1', '#FF9DA7', '#9C755F', '#BAB0AB']
sns.set_palette("pastel")
plt.rcParams['figure.figsize'] = (9, 6)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("NBA_Train.csv")
df


# ## Data Preprocessing

# ### Useless Columns

# In[4]:


df.drop(columns="Train_ID", inplace=True)


# ### Missing value

# In[5]:


df.isnull().sum()


# ### Outliers

# In[6]:


def _scatter(df, feature, target):
    """

    """
    # plt.figure(figsize=(16,8))
    plt.scatter(df[feature], df[target])
    plt.xlabel(feature)
    plt.ylabel(target)

    items = [feature, " vs ", target]
    title = ''.join(items)
    plt.title(title)


def subplot_scatter(df, target):
    """
    Plot scatter figures of each column in the dataFrame. 
    Args:
        df: pandas.DataFrame 
            DataFrame input.

        target: pandas.Series()
            Target column.


    """

    plt.figure(figsize=(30, 90))
    num_subplot = len(df.columns.drop(target))
    for i, col in enumerate(df.columns.drop(target)):
        plt.subplot(num_subplot//2 + 1, 2, i+1)
        _scatter(df, col, target)


def subplot_box(df):
    """

    """

    plt.figure(figsize=(30, 90))
    num_subplot = len(df.columns)
    for i, col in enumerate(df.columns):
        plt.subplot(num_subplot//2 + 1, 2, i+1)
        sns.set_palette("pastel")
        sns.boxplot(x=df[col])


def zscore_drop_missing(df, col_list, THRESHOLD=3):
    """
    """
    for col in col_list:
        z = np.abs(stats.zscore(df[col]))
        df = df[(z < THRESHOLD)]

    df.index = range(len(df))
    return df



# In[7]:


col_category = ["POSITION", "TEAM"]
col_number = df.columns.drop(["TEAM", "POSITION","SALARY"])


# subplot_scatter(df, "SALARY")

with sns.color_palette(crayon):
    plot_regressions(df[col_number], df["SALARY"])
    plt.show()


# In[8]:


subplot_box(df[col_number])


# In[9]:


#"SALARY"
df = zscore_drop_missing(df, [col_number], THRESHOLD=3.5)


# In[10]:


# subplot_scatter(df, "SALARY")
with sns.color_palette(crayon):
    plot_regressions(df[col_number], df["SALARY"])
    plt.show()


# In[11]:


subplot_box(df[col_number])


# ### Normalization

# In[12]:


normalized  = lambda x: (x - x.min()) * (1-0) / (x.max()-x.min())

normalized(df[col_number])


# ## Exploratory data analysis (EDA)

# In[13]:


def get_status(x):
    return pd.DataFrame([x.count(), x.mean(), x.std(), x.min(), x.quantile(.25), x.quantile(.5), x.quantile(.75), x.max(), x.median(), x.mad(), x.var(), x.std(), x.skew(), x.kurt(), ],
                        index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'median', 'mad', 'var', 'std', 'skew', 'kurt', ]).round(3)


def plot_distributed(series):
    
    sns.distplot(series, fit=norm, bins=10, color='cornflowerblue')
    (mu, sigma) = norm.fit(series)
    plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=${:.2f})'.format(
        mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('Distribution of '+series.name)

def subplot_distributed(df):
    """

    """
    plt.figure(figsize=(30, 90))
    num_subplot = len(df.columns)
    for i, col in enumerate(df.columns):
        plt.subplot(num_subplot//2 + 1, 2, i+1)
        sns.set_palette("pastel")
        
        plot_distributed(df[col])

def plot_prob(series):
    stats.probplot(series,plot=plt)
    

def subplot_prob(df):
    """

    """
    plt.figure(figsize=(30, 90))
    num_subplot = len(df.columns)
    for i, col in enumerate(df.columns):
        plt.subplot(num_subplot//2 + 1, 2, i+1)
        sns.set_palette("pastel")
        plot_prob(df[col])
        


# ### Target Variable

# In[14]:


get_status(df["SALARY"])


# In[15]:


plot_distributed(df["SALARY"])


# In[16]:


plot_prob(df["SALARY"])


# In[17]:


get_status(np.log(df["SALARY"]))


# In[18]:


plot_distributed(np.log(df["SALARY"]))


# In[19]:


plot_prob(np.log(df["SALARY"]))


# ### Numerical Variables

# In[20]:


df_eda = normalized(df[col_number])
get_status(df_eda)


# In[21]:


subplot_distributed(df_eda)


# In[22]:


subplot_prob(df_eda)


# In[23]:


skew_high_cols = ["ORB","AST","BLK","TOV","OWS","DWS","WS"]
for col in skew_high_cols:
    df_eda[col] = np.sqrt(df_eda[col])
    
skew_low_log = ["Games"]
for col in skew_low_log:
    df_eda[col] = df_eda[col] ** 2


# In[24]:


get_status(df_eda)


# In[25]:


subplot_distributed(df_eda)


# In[26]:


subplot_prob(df_eda)


# In[27]:


df_eda = pd.concat([df_eda,np.log(df["SALARY"])],axis=1)
plt.figure(figsize = (10,8))
sns.heatmap(df_eda.corr(), vmin = 0, vmax = 1, cmap = 'Reds')
plt.title('Correlation Among Variables', fontsize = 16)


# In[28]:


print(abs(df_eda.corr()["SALARY"]).sort_values(ascending=False))


# In[29]:


sns.pairplot(df_eda[["Games","Minutes"]])


# In[30]:


with sns.color_palette(crayon):
    plot_regressions(df_eda[["Games","Minutes"]], df_eda["SALARY"])
    plt.show()


# In[31]:


sns.pairplot(df_eda[["ORB","DRB","TRB"]])


# In[32]:


with sns.color_palette(crayon):
    plot_regressions(df_eda[["ORB","DRB","TRB"]], df_eda["SALARY"])
    plt.show()


# In[33]:


sns.pairplot(df_eda[["OWS","DWS","WS"]])


# In[34]:


with sns.color_palette(crayon):
    plot_regressions(df_eda[["ORB","DRB","TRB"]], df_eda["SALARY"])
    plt.show()


# ### Category Variables

# In[35]:


plt.figure(figsize=(20,10))
sns.set_palette("pastel")
sns.boxplot(x = 'TEAM', y = 'SALARY', data = df)
sns.despine()
plt.title('Correlation between SALARY and TEAM', fontsize = 16)
plt.xticks(rotation=270)
plt.show()


# In[36]:


sns.set_palette("pastel")
sns.boxplot(x = 'POSITION', y = 'SALARY', data = df)
sns.despine()
plt.title('Correlation between SALARY and TEAM', fontsize = 16)
plt.show()


# In[37]:


def fe_categorical_var(df, cols):
    d = pd.DataFrame()
    for col in cols:
        dummies = pd.get_dummies(
            df[col], prefix_sep="_", drop_first=True, prefix=col)
        #df.drop(columns=col, inplace=True)
        d = pd.concat([d, dummies], axis=1)

    return d


temp = pd.concat([fe_categorical_var(df, col_category),
                  df["SALARY"]], axis=1)


# In[38]:


plt.figure(figsize=(10, 8))
sns.heatmap(temp.corr(), vmin=0, vmax=1, cmap='Reds')
plt.title('Correlation Among Variables', fontsize=16)


# In[39]:


print(abs(temp.corr()["SALARY"]).sort_values(ascending=False))


# ## Feature engineering

# In[40]:


def fe_target_var(df):
    return pd.DataFrame(np.log(df["SALARY"]))


def fe_numerical_var(df):

    df_number = normalized(df[col_number])
    for col in skew_high_cols:
        df_number[col] = np.sqrt(df_number[col])

    for col in skew_low_log:
        df_number[col] = df_number[col] ** 2

    df_number.drop(columns=["Games","TRB", "ORB",
                            "DWS", "OWS"], axis=1, inplace=True)

    return df_number


def fe_categorical_var(df, cols):
    d = pd.DataFrame()
    for col in cols:
        dummies = pd.get_dummies(
            df[col], prefix_sep="_", drop_first=True, prefix=col)
        #df.drop(columns=col, inplace=True)
        d = pd.concat([d, dummies], axis=1)

    return d


def feature_eng(df):
    return pd.concat(
        [fe_target_var(df), fe_numerical_var(df), fe_categorical_var(df, col_category)], axis=1)


# In[41]:


df_engineered = feature_eng(df)
df_engineered


# In[42]:


# selected_features = []


# ## Modelling

# In[43]:


X_train = df_engineered.iloc[:,1:]
y_train = df_engineered["SALARY"]


# In[44]:


# df_engineered.columns


# In[45]:


df_test = feature_eng(pd.read_csv("NBA_Test.csv"))
df_test['TEAM_Houston Rockets'] = 0
df_test = df_test[df_engineered.columns]

X_test = df_test.iloc[:,1:]
y_test = df_test["SALARY"]


# ### Lasso

# In[46]:


alpha = list(np.logspace(-4, -.5, 30))

lasso = LassoCV(cv=5, random_state=0, alphas=alpha)

lasso.fit(X_train, y_train)


# In[47]:


lasso.alpha_


# In[48]:


lasso.coef_


# In[49]:


y_pred = lasso.predict(X_test)


# In[50]:


np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_pred)))


# In[51]:


def plot_resid(y_test, y_pred):
    """
    
    """
    
    resid = y_test-y_pred
    
    plt.figure(figsize = (16,9))
    plt.scatter(y_pred,resid)
    plt.axhline(0,0,1, color="g", ls="--")
    plt.xlabel("Fitted")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.savefig("Residual plot ols")
    plt.show()
    
plot_resid(y_test, y_pred)


# ### Gradient Boosting

# In[52]:


tuning_parameters = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [250, 500, 750, 1000, 1500],
    'max_depth': [2, 3, 4],
    'subsample': [0.6, 0.8, 1.0]
}

# Using GridSearchCV would be too slow. Increase the number of iterations to explore more hyperparameter combinations.
gb = RandomizedSearchCV(GradientBoostingRegressor(), tuning_parameters, n_iter=1, cv=10, return_train_score=False, n_jobs=4)
gb.fit(X_train, y_train)

print('Best parameters found by randomised search:', gb.best_params_, '\n')


# In[53]:


gb_best = gb.best_estimator_


# In[54]:


from statlearning import plot_feature_importance

plot_feature_importance(gb.best_estimator_, list(X_train))
plt.show()


# In[55]:


y_pred= gb.predict(X_test)


# In[56]:


np.sqrt(mean_squared_error(np.exp(y_test),np.exp(y_pred)))


# In[57]:


toc = time.time()
print("Training time: {0:.4f}s".format(toc - tic))

