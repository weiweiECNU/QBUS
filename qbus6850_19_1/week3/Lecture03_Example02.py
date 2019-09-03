#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:04:56 2018

@author: jbgao, revised from
https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
"""
from numpy import array 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# define One Feature for fealing of temperature.  There are 3 three value. 
# This is a categorical feature with K = 3 (different categories)
# We have ten cases of observations. We now organize them in the data shape (row means
# cases, column means feature (1 column here))
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
values.reshape(len(values), 1)
print(values)

# First step, we are going to use integer label to represent each value
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
# Check the fitted code to see which integer is for which of 'cold', 'warm', 'hot'
# They could be 'cold' -> 0; 'warm' -> 1; 'hot' -> 2;  or 
# they could be 'warm' -> 0; 'hot' -> 1; 'cold' -> 2; or others
print(integer_encoded)
# We make sure in our data shape (row as cases column as features, one feature here)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)


# With the integer label code, we can convert them into one-hot code
# binary encode

# Define the encoder, as this is a small data set, we dont need sparse representation
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# In one-hot code, in terms of data shape, we have increase the number of feature
# from 1 to 3. Each new feature corresponds to one categorical value of the original 
# feature

"""
You can use OneHotEncoder to code multiple categorical features in one go.
Look at the following example
"""

data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
"""
The above data contains 4 cases with 3 categorical features in the integer label form
For example, the first row [0, 0, 3] means that for the first case the value for 
the first feature is 0, the second feature is also 0, while the third feature 
value is 3.   Accoding to values, most likely the first feature has 2 different values;
the second feature has 3 different values, while the third feature has 4 differnet values
"""

# Define a encoder
enc = OneHotEncoder()
# Fit to the data,  
enc.fit(data)  

#the encoder automatically figures out the number of values for each feature.
# You can check this by  
enc.n_values_

# We can find a new case's one-hot code
enc.transform([[0, 1, 2]]).toarray() 
# How many columns in hot-one code do we have for this case?    2 + 3 + 4

# Note: pandas and keras also offer functionality to find one-hot code, see
# http://pbpython.com/categorical-encoding.html