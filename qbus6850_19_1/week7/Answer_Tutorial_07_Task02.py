#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:21:43 2018

@author: Professor Junbin Gao
"""
from sklearn import tree
#import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image, display
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# For extracting decision rules
from sklearn.tree import _tree 
def tree_to_code(tree, feature_names):

	tree_ = tree.tree_

	feature_name = [
		feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
		for i in tree_.feature
	]
	print ("def tree({}):".format(", ".join(feature_names)))

	def recurse(node, depth):
		indent = "  " * depth
		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = feature_name[node]
			threshold = tree_.threshold[node]
			print ("{}if {} <= {}:".format(indent, name, threshold))
			recurse(tree_.children_left[node], depth + 1)
			print ("{}else:  # if {} > {}".format(indent, name, threshold))
			recurse(tree_.children_right[node], depth + 1)
		else:
			print ("{}return {}".format(indent, tree_.value[node]))

	recurse(0, 1)
# End of extracting decision rules

#Load the dataset and remove columns that are all NaN values. 
#These columns are not neccesary and would cause an error with our 
#DecisionTreeClassifier later.
loans = pd.read_csv("LoanStats3a_2.csv")

# Remove NaN columns
loans = loans.dropna(axis = 1, how = "all")

# Some columns are not imformative or we cannot possibly create dummy 
# variables for them. So we also remove them.
# Here I select the useful columns as the set subtraction of 
# GOOD COLUMNS = (EVERY COLUMN - BAD COLUMNS). Then grab the good columns.

bad_cols = ['id', 'emp_title', 'emp_length', 'addr_state', 'next_pymnt_d', 'earliest_cr_line', 'last_pymnt_d', 'desc', 'revol_util', 'last_credit_pull_d', 'title', 'pymnt_plan', 'int_rate', 'zip_code', 'sub_grade', 'purpose', 'grade', 'initial_list_status', 'issue_d', 'verification_status', 'application_type']
final_columns = list(set(loans.columns) - set(bad_cols))

final = loans.loc[:, final_columns]

#For our decision tree to work we can't have NaN values in our data. 
#So I remove rows that contain any NaN value.
#Be warned: this removes a lot of data.

final = final.dropna(axis = 0, how = "any")

#Seperate the data into features and target
X = final.loc[:, final.columns != 'loan_status']
X = pd.get_dummies(X)
feature_names = X.columns

y = final['loan_status']

#Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Find the optimal tree.
#First we set up the parameters to search over including the tree depth 
# and criterion. This creates the cv_clf object.

tree_depths = np.arange(1, 11, 1)

param_grid = {'max_depth': tree_depths, 'criterion': ['gini', 'entropy'] }

cv_clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv = 5)

# To actually perform the Cross Validation you need to call cv_clf.fit().
cv_clf.fit(X_train, y_train)

#To get the DecisionTreeClassifier from the GridSearchCV object 
#we can access the best_estimator attribute
clf = cv_clf.best_estimator_
print(clf)

# extract true rules
tree_to_code(clf, feature_names)


# If you have installed Graphviz and Pydotplus, you can uncomment this section
"""
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image, display
import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

f = StringIO()

tree.export_graphviz(clf, out_file=f)

graph = pydotplus.graph_from_dot_data(f.getvalue())

display(Image(graph.create_png()))
"""
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

 