"""
Created on Wed Mar  28 09:34:07 2017

@author: Dr Chao Wang
Revised by: Professor Junbin Gao
"""
from sklearn import tree
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Load the dataset, please view the csv file by using excel to 
# see what information is in the file
purchase_df = pd.read_csv("Lecture6_Data.csv")

# The three predictors are Income, Education and Marital Status
# which are categorical, so we use hot-one code for them
purchase_df_x = pd.get_dummies(purchase_df.iloc[:, 1:-1])

# Label is categorical too. We also need transform it to 
# class label code.  If there are 3 classes, the code would be
# 0, 1, or 2.
le = LabelEncoder()
y_train = le.fit_transform(purchase_df['Purchase'])

# Which of these features gives us the best split?
purchase_df_xy = purchase_df_x
purchase_df_xy['y'] = y_train   #Check what this variable contains

feature_names = purchase_df_x.columns

# using the Entropy as measurement of impurity
# We assume data D in argument "rows" variable, the last column
# is label code

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

# For each possible split subset, we calculate its entropy
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

#################################################
# using sklearn 

purchase_df_x = pd.get_dummies(purchase_df.iloc[:, 1:-1])

clf = tree.DecisionTreeClassifier(max_depth = 1)
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 1)

clf = clf.fit(purchase_df_x, y_train)
# sklearn tree class assignment

print(clf.tree_.value[1:3])
clf.feature_importances_ 

most_likely_class = clf.predict(purchase_df_x.iloc[1,:])
print(most_likely_class)

# visualising trees
# Unless you have installed pydotplus, you can uncomment this
"""
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image, display

# Create a string buffer to write to (a fake text file)
f = StringIO()
# Write the tree description data to the file
tree.export_graphviz(clf, out_file=f)
# Produce a visulation from the file
graph = pydotplus.graph_from_dot_data(f.getvalue())
display(Image(graph.create_png()))



import os     
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

"""

































