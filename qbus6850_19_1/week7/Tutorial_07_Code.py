#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:21:43 2018

@author: Professor Junbin Gao
"""
from __future__ import division
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

# Task 1
purchase_df = pd.read_csv("Lecture6_Data.csv")

purchase_df_x = pd.get_dummies(purchase_df.iloc[:, 1:-1])

le = LabelEncoder()
y_train = le.fit_transform(purchase_df['Purchase'])


purchase_df_xy = purchase_df_x
purchase_df_xy['y'] = y_train 

feature_names = purchase_df_x.columns

def h_function_entropy (rows):
    classes = np.unique(rows.iloc[:, -1])
    entropy_all = []
    N_node = len(rows)
    
    for k in classes:
        # calcualte the proportion p(x) for each class. Now we    
        # have only two classes. 1 and 0.
        prop_temp= len( rows[purchase_df_xy.iloc[:, -1] == k] )/N_node
        
        entropy_temp = -(prop_temp) *np.log2(prop_temp)
        entropy_all.append(entropy_temp)
    
    entropy_all = np.array(entropy_all)
    # calculate the entropy for each subset
    entropy_final= np.sum(entropy_all)
    return entropy_final

loss_list = []

# For each possible split
for i in range(purchase_df_xy.shape[1] - 1):
    
    # Find observations falling to the left
    left_x = purchase_df_xy[purchase_df_xy.iloc[:, i] == 0]
    
    # Add your code to find observations falling to the right
    right_x = purchase_df_xy[purchase_df_xy.iloc[:, i] == 1]
    
    # Calculate the weighted average based on sample size as the loss of this split
    N_branch = len(left_x) + len(right_x)
    # h_function_entropy(left_x) and h_function_entropy(right_x) are entropies of subsets
    loss = (len(left_x) / N_branch) * h_function_entropy(left_x) + (len(right_x) / N_branch) * h_function_entropy(right_x)

    loss_list.append(loss) 
    
feature_index = np.argmin(loss_list)

final_left_rows = purchase_df_xy[purchase_df_xy.iloc[:, feature_index] == 0]
final_right_rows = purchase_df_xy[purchase_df_xy.iloc[:, feature_index] == 1]

n_classes = len(np.unique(purchase_df_xy.iloc[:,-1]))
value_left = np.zeros(n_classes)
value_right = np.zeros(n_classes)

for i in final_left_rows.iloc[:, -1]:
    value_left[i] = value_left[i] + 1 

for i in final_right_rows.iloc[:, -1]:
    value_right[i] = value_right[i] + 1
    
print("Left node: {0}".format(value_left))
print("Right node: {0}".format(value_right))



# Task 2
purchase_df_x = pd.get_dummies(purchase_df.iloc[:, 1:-1])
clf = tree.DecisionTreeClassifier(max_depth = 1)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 1)
clf = clf.fit(purchase_df_x, y_train)

# sklearn tree class assignment
print(clf.tree_.value[1:3])
clf.feature_importances_

# Show our results 
print("Left values: {0}".format(value_left))
print("Right values: {0}".format(value_right))

most_likely_class = clf.predict(purchase_df_x.iloc[1,:].values.reshape(1,-1))
print(most_likely_class)

most_likely_class = clf.predict(purchase_df_x.iloc[2:6,:])
print(most_likely_class) 

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


# extract true rules
tree_to_code(clf, feature_names)


# If you have installed Graphviz and Pydotplus, you can uncomment this section
"""
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image, display
import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Create a string buffer to write to (a fake text file)
f = StringIO()

# Write the tree description data to the file
tree.export_graphviz(clf, out_file=f, proportion=False)

# Produce a visulation from the file
graph = pydotplus.graph_from_dot_data(f.getvalue())

display(Image(graph.create_png()))
"""



# Task 3
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

default = pd.read_csv("default.csv")

default = pd.get_dummies(default)

y = default['default']

X = default.iloc[:, 1:]
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

print("The total number of default is {}.".format(np.sum(y == 1)))
print("The total number of non-default is {}.".format(np.sum(y == 0)))
 
default_clf = tree.DecisionTreeClassifier(max_depth = 5, class_weight = 'balanced')

default_clf.fit(X_train, y_train)

print(default_clf.tree_.value[:, 0][0:2])
 
# extract true rules
tree_to_code(default_clf, feature_names)

y_pred = default_clf.predict(X_test)

print("Number of defaults in test set: {0}".format(sum(y_test)))

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Create Tree Graph
# If you have installed Graphviz and Pydotplus, you can uncomment this section
"""
# Create a string buffer to write to (a fake text file)
f = StringIO()

# Write the tree description data to the file
tree.export_graphviz(default_clf, out_file=f, proportion=True)

# Produce a visulation from the file
graph = pydotplus.graph_from_dot_data(f.getvalue())

display(Image(graph.create_png())) 
"""