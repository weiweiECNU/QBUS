from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# http://archive.ics.uci.edu/ml/datasets/communities+and+crime

crime = pd.read_csv('communities.csv', header=None, na_values=['?'])

# Delete the first 5 columns
crime = crime.iloc[:, 5:]

# Remove rows with missing entries
crime.dropna(inplace=True)

# Get the features X and target/response y
X = crime.iloc[:, :-1]
t = crime.iloc[:, -1]

# Split data into train and test sets
X_train, X_test, t_train, t_test = train_test_split(X, t, random_state=1)

#%%
# Fit OLS model and calculate loss values over test data
lm = LinearRegression()
lm.fit(X_train, t_train)
preds_ols = lm.predict(X_test)
print("OLS: {0}".format(mean_squared_error(t_test, preds_ols)/2))

#%% Ridge
alpha_range = 10.**np.arange(-2, 3)

rregcv = RidgeCV(
    normalize=True, scoring='neg_mean_squared_error', alphas=alpha_range)
rregcv.fit(X_train, t_train)
preds_ridge = rregcv.predict(X_test)
print("RIDGE: {0}".format(mean_squared_error(t_test, preds_ridge)/2))

#%%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X_txt = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names() == (
    ['and', 'document', 'first', 'is', 'one',
     'second', 'the', 'third', 'this'])

# X is the BOW feature of X
print(X_txt.toarray())

# the value is vocab id
print(vectorizer.vocabulary_)

# Unseen words are ignore
vectorizer.transform(['Something completely new.']).toarray()

#%% Bigram

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print(analyze('believe or not a b c d e'))

X_txt_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print(X_txt_2)

#%% TD IDF
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]
          ]
tfidf = transformer.fit_transform(counts)
print(tfidf)
print(tfidf.toarray())

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
print(corpus)
X_txt_3 = vectorizer.fit_transform(corpus)
print(X_txt_3.toarray())

#%% Feature selection LASSO/Ridge
from sklearn.linear_model import LassoCV, ElasticNetCV

X = crime.iloc[:, :-1]
y = crime.iloc[:, -1]
X_train, X_test, t_train, t_test = train_test_split(X, t, random_state=1)

lascv = LassoCV()
lascv.fit(X_train, t_train)

preds_lassocv = lascv.predict(X_test)
print("LASSO: {0}".format(mean_squared_error(t_test, preds_lassocv)/2))
print("LASSO Lambda: {0}".format(lascv.alpha_))

#%% Elastic

elascv = ElasticNetCV(max_iter=10000)
elascv.fit(X_train, t_train)

preds_elascv = elascv.predict(X_test)
print("Elastic Net: {0}".format(mean_squared_error(t_test, preds_elascv)/2))
print("Elastic Net Lambda: {0}".format(elascv.alpha_))


#%%Feature
columns = X.columns.values

lasso_cols = columns[np.nonzero(lascv.coef_)]
print("LASSO Features: {0}".format(lasso_cols))

elas_cols = columns[np.nonzero(elascv.coef_)]
print("Elastic Features: {0}".format(elas_cols))

#%% Logistic regression
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('Default.csv')

# Convert the student category column to Boolean values
df.student = np.where(df.student == 'Yes', 1, 0)

X = df[['balance']]
y = df[['default']]

X_train, X_val, t_train, t_val = train_test_split(
    X, y, test_size=0.3, random_state=1)


log_res = LogisticRegression()
log_res.fit(X_train, t_train)

#%%
# Predict probabilities of default.
#
# predict_proba() returns the probabilities of an observation belonging to each class. This is computed from the logistic regression function (see above).
#
# predict() is dependant on predict_proba(). predict() returns the class assignment based on the proability and the decision boundary. In other words predict returns the most likely class i.e. the class with greatest probability or probability > 50%.

prob = log_res.predict_proba(pd.DataFrame({'balance': [1200, 2500]}))
print(prob)
print("Probability of default with Balance of 1200: {0:.2f}%".format(
    prob[0, 1] * 100))
print("Probability of default with Balance of 2500: {0:.2f}%".format(
    prob[1, 1] * 100))

outcome = log_res.predict(pd.DataFrame({'balance': [1200, 2500]}))
print("Assigned class with Balance of 1200: {0}".format(outcome[0]))
print("Assigned class with Balance of 2500: {0}".format(outcome[1]))

#%%
# We can evaluate classification accuracy using confusion matrix (to be explanined in week 4 lecture) and the classification report

from sklearn.metrics import confusion_matrix
pred_log = log_res.predict(X_val)
print(confusion_matrix(t_val, pred_log))
from sklearn.metrics import classification_report
print(classification_report(t_val, pred_log, digits=3))
