#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:28:55 2018

@author: steve
"""

"""
Importing Libraries

Numpy: mathematics and linear algebra library
Pandas: data handling
Pyplot: simple plotting

"""
#%%
#Next week repeating
##worth repeating next week


'''
a = np.array([1, 2, 3])
a = np.array([[1, 2, 3]])

A = np.matrix([1, 2, 3])

ac = a[:,1]
ac = A[:,1]
print(ac)
'''

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Vectors

- Creation
- Special vectors: 0, 1, abitrary scalar, linspace, arange
- Scalar Multiplication
- Transpose
- Check equality
- Sum
- Linear Combinations
- Inner product (scalar result), outer product (matrix result)
- Norms, Length
- Orthogonality
"""

##worth repeating next week
#a = np.array([1, 2, 3])


a = np.array([1, 2, 3] )

A = np.matrix([1, 2, 3])


print(a.shape)
print(A.shape)

# Product of arrays is elementwise
print(a * a)

# Use dot product on arrays if you want normal product
print(np.dot(a,a))

# Note that the type of the object is inferred
# In this case we have used integers
# Lets try with floating point numbers (decimals)

print(a.dtype)
type(a)

#%%
# How to initialize

c = np.array([1.0, 2.0, 3.0])

print(c.dtype)

# Some useful shortcuts for special vectors

# Length 10 zero vector
zeros = np.zeros((10,1))

# Length 5 ones vector
ones = np.ones((5,1))

# You can create a single valued vector as the product
# of a vector of ones and your desired value
twos = ones * 2

# 11 numbers from 0 to 100, evenly spaced
lin_spaced = np.linspace(0, 100, 11)

# All numbers from 0 up to (not including) 100 with gaps of 2
aranged = np.arange(0, 100, 2)

# Transpose, only works on arrays with fully specified dimensions
A_transposed = A.transpose()

# Check equality
print(np.array_equal(A, A))
print(np.array_equal(A, A_transposed))

# You may wish to check equality up to a tolerance
# This is useful since floating points aren't perfect
print(np.allclose(A, A, rtol=1e-05))

# Sums of arrays
c = a + a

# Linear Combinations
d = 3*a * 1.5*c

# 2-norm of vector (length)
norm2 = np.linalg.norm(a, ord=2)
print(norm2)

# Orthogonality
# Generate an orthogonal vector and test it
x = [1, 2, 3]
y = [4, 5, 6]
orthog = np.cross(x, y)

# If vectors are orthog then their inner product is 0
print(np.dot(x, orthog))

#%%%
"""
Matrices

- Creation
- Transpose
- Special matrices: 0, identity, abitrary scalar, diag
- Scalar Multiplication
- Sum
- Matrix Product
- Rank
- Determinant
- Trace
- Inverse
"""

b = np.array([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9]])

B = np.matrix([[1, 4, 7],
              [2, 5, 8],
              [3, 6, 9]])


print(B)

# Multiplication of matrices is as normal
print(B * B)

# Scalar Product, Sum and Transpose are standard
print(5 * B)

print(B + B)

print(B.transpose())

#%%
# Some useful shortcuts for special matrices
m_zeros = np.zeros((10,10))
m_ones = np.ones((5,5))
m_twos = m_ones * 2

# Identity matrix
eye = np.identity(3)


diag_elements = np.diag(B)

# Create a matrix with values along the diagonal
m_diag = np.diag([1,2,3])


# Matrix Rank i.e. n linearly independent cols or rows
print(np.linalg.matrix_rank(B))

# Sum of diagonal entries
print(np.trace(B))

# Determinant
e = np.array([[1, 2], 
              [3, 4]])
print(np.linalg.det(e))


#%%
"""
Loading Data/Pandas

- Pandas
- Indexing: Named Columns, Numbered Columns, Rows, iloc etc
- Searching
- Cleaning and deleting
- Dataframe vs Series

"""

# Load the drinks Comma Seperated Value file
# Store in DataFrame called drinks
drinks = pd.read_csv('drinks.csv')

# Display information about the DataFrame
drinks.dtypes
drinks.info()

# Display summary statistics of the DataFrame
drinks.describe()

# Extract the beer_servings column as a Series 
drinks['beer_servings']

# Summarize only the beer_servings Series
drinks['beer_servings'].describe()   

# Get the mean of beer_servings Series
drinks['beer_servings'].mean()         

# Get the sub DataFrame where the contitent is Europe
euro_frame = drinks[drinks['continent'] == 'EU']

# Get the mean of beer_servings
euro_frame['beer_servings'].mean()


#Get the coutry with 'wine_servings' greater than 300
heavywind = drinks[drinks['wine_servings'] > 300]

# Get European countries with 'wine_servings' greater than 300
euro_heavywine = drinks[(drinks['continent'] == 'EU') & (drinks['wine_servings'] > 300)]
#%%

'''
Handling Missing Values
'''

# Load data without filtering missing entries
# Missing entries will be replaced with NaN values

default_na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA', 'NULL', 'NaN', 'nan']

our_na_values = default_na_values
our_na_values.remove('NA')

drinks_dirty = pd.read_csv('drinks_corrupt.csv',keep_default_na = False, na_values = our_na_values)

# See the number of NaN values per column
drinks_dirty.isnull().sum()

# Remove all rows from drinks with any missing columns
drinks_clean = drinks_dirty.dropna()

# Remove rows only if every column has a missing entry
drinks_clean_byrows = drinks_dirty.dropna(how='all') 

# Do an inplace removal. No need to create a new variable.
drinks_dirty.dropna(inplace = True)

# Replacement
drinks = pd.read_csv('drinks.csv')

# Replace missing continent values with 'NA'
drinks['continent'].fillna(value='NA', inplace=True)

# Create new columns as a function of existing columns
drinks['total_servings'] = drinks.beer_servings + drinks.spirit_servings + drinks.wine_servings
drinks['alcohol_mL'] = drinks.total_litres_of_pure_alcohol * 1000

# Check changes
most_heavy = drinks.sort_values(by='total_litres_of_pure_alcohol').tail(10)
most_heavy = drinks.sort_values(by='total_litres_of_pure_alcohol', ascending = False).head(10)

#%%


'''
Renaming
'''

# Columns can be easily renamed
drinks.rename(columns={'total_litres_of_pure_alcohol':'alcohol_litres'}, inplace=True)

'''
Deleting Data
'''

# Columns can be easily delated
drinks_wout_ml = drinks.drop(['alcohol_litres'], axis=1)

#%%
"""
Plotting/Pyplot

- Figures as a canvas
- Plot types
- Labels
- Titles
- Legends
"""

# Pyplot draws each plot on a Figure
# Think of a Figure as a blank canvas
# By default Pyplot will draw to the last Figure you created
plt.figure()

# Bar Plot
# Plot the number of beer servings of heavy wine drinking european countries
ind = np.arange(len(euro_heavywine))

plt.bar(ind, euro_heavywine['beer_servings'])

plt.xticks(ind, euro_heavywine['country'])

plt.xlabel("Country")

plt.ylabel("Beer Servings")

plt.title("Beer Servings of Heavy Wine Drinking EU Countries")

# Line Plot
# Plot the total servings of top drinkers in desceding order
plt.figure()

# Get the 10 countries with highest total servings
top_drinkers = drinks.sort_values(by='total_servings', ascending=False).head(10)

# Line plot with a label and custom colour
# Other optional parameters include linestyle and markerstyle
plt.plot(np.arange(0,10,1), top_drinkers['total_servings'], label="Total Servings", color="red")

plt.xlabel("Drinking Rank")

plt.ylabel("Total Servings")

plt.title("Drinking Rank vs Total Servings")

# Activate the legend using label information
plt.legend()


