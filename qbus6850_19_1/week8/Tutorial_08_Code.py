#Task1
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

bank_df = pd.read_csv("bank.csv")

X = bank_df.iloc[:, 0:-1]
y = bank_df['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


clf = ensemble.RandomForestClassifier(class_weight = 'balanced')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

param_grid = {'n_estimators': np.arange(1,50,10),
              'max_depth': np.arange(1,20,1), } 

clf_cv = GridSearchCV(ensemble.RandomForestClassifier(class_weight = 'balanced'), param_grid)

print("Running....")
tic = time.time()
clf_cv.fit(X_train, y_train)

toc = time.time()
print("Training time: {0}s".format(toc - tic))


clf = clf_cv.best_estimator_
print(clf)


y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#%%
#Task 2
clf = ensemble.AdaBoostClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#%%
#Task 3
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
    
#%%
#Task 4
    
    
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()

corpus = ['This is the first document.',
          'This is the second second document.']

X = count_vectorizer.fit_transform(corpus)
    
    
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))

tfidf = tfidf_vectorizer.fit_transform(corpus)

print(tfidf_vectorizer.get_feature_names())

print(tfidf)


from sklearn.datasets import fetch_20newsgroups

categories = None
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)


X_train = data_train.data
y_train = data_train.target

X_test = data_test.data
y_test = data_test.target

# tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

print("Fitting and Transforming")
usenet_tfidf = tfidf_vectorizer.fit_transform(X_train)
print("Done")

usenet_clf = ensemble.RandomForestClassifier(n_estimators=50)

print("Training....")
usenet_clf.fit(usenet_tfidf, y_train)
print("Training completed.")

y_pred = usenet_clf.predict(tfidf_vectorizer.transform(X_test))

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
