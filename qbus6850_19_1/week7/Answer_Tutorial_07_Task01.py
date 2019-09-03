#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:21:43 2018

@author: Professor Junbin Gao
"""
from __future__ import division
from sklearn.datasets import load_iris
from sklearn import tree
#import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image, display
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

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

#Load and pre-process the data. Convert class labels to numeric class indices. To be updated.
purchase_df = pd.read_csv("Lecture6_Data.csv")

purchase_df_x = pd.get_dummies(purchase_df.iloc[:, 1:-1])

le = LabelEncoder()
y_train = le.fit_transform(purchase_df['Purchase'])

# Which of these features gives us the best split?
purchase_df_xy = purchase_df_x
purchase_df_xy['y'] = y_train

feature_names = purchase_df_x.columns

#Define our classification critera. I have chosen the Gini metric
def h_function_gini (rows):
    classes = np.unique(rows.iloc[:, -1])
    gini_all = []
    N_node = len(rows)
    
    for k in classes:
        # calcualte the proportion p(x) for each class
        prop_temp= len( rows[purchase_df_xy.iloc[:, -1] == k] )/ N_node
        gini_temp = (prop_temp) *(1-prop_temp)
        gini_all.append(gini_temp)
    
    gini_all = np.array(gini_all)
    gini_final= np.sum(gini_all)
    return gini_final

# For each possible branch in our tree compute the loss (or impurity). 
# We need to first split the data according to the category then compute 
# the loss value.
loss_list = []

# For each possible split
for i in range(purchase_df_xy.shape[1] - 1):
    
    # Find observations falling to the left (0)
    left_x = purchase_df_xy[purchase_df_xy.iloc[:, i] == 0]
    
    # Find observations falling to the right(1)
    right_x = purchase_df_xy[purchase_df_xy.iloc[:, i] == 1]

    # Calculate the gini
    N_branch = len(left_x) + len(right_x)
    loss = (len(left_x) / N_branch) * h_function_gini(left_x) + (len(right_x) / N_branch) * h_function_gini(right_x)

    loss_list.append(loss)

#To find the minimum impurity split you can use argmin
feature_index = np.argmin(loss_list)

# Split the data according to the optimal split
final_left_rows = purchase_df_xy[purchase_df_xy.iloc[:, feature_index] == 0]
final_right_rows = purchase_df_xy[purchase_df_xy.iloc[:, feature_index] == 1]

#Calculate the number of observations from each class that was assigned to the left and right leaves
n_classes = len(np.unique(purchase_df_xy.iloc[:,-1]))
value_left = np.zeros(n_classes)
value_right = np.zeros(n_classes)

for i in final_left_rows.iloc[:, -1]:
    value_left[i] = value_left[i] + 1 

for i in final_right_rows.iloc[:, -1]:
    value_right[i] = value_right[i] + 1
    
print("Left node: {0}".format(value_left))
print("Right node: {0}".format(value_right))

#Let's compare to the SKLearn classifier.
purchase_df_x = pd.get_dummies(purchase_df.iloc[:, 1:-1])

clf = tree.DecisionTreeClassifier(max_depth = 1)

clf = clf.fit(purchase_df_x, y_train)

# sklearn tree class assignment
print(clf.tree_.value[1:3])

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
tree.export_graphviz(clf, out_file=f)

# Produce a visulation from the file
graph = pydotplus.graph_from_dot_data(f.getvalue())

# Write visualisation to image file 
graph.write_png("dtree_gini.png")

#display(Image(filename="dtree2.png"))

display(Image(graph.create_png()))

"""
 
 