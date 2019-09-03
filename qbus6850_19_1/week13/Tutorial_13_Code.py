# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 16:54:11 2017

@author: cwan6954
"""
# https://pypi.python.org/pypi/scikit-surprise
import random

from surprise import SVD
from surprise import KNNBasic
from surprise import BaselineOnly
from surprise import Dataset
from surprise import accuracy
from surprise import GridSearch

# Load the full dataset.
data = Dataset.load_builtin('ml-100k')


#Now we will do a grid search using cross validation to find the optimal parameters for the SVD method.
#To specify the number of cross validation folds to use in the evaluate() function you need to use the split() function of the dataset.

data.split(n_folds=3)

#################
# visualise the data
# https://medium.com/@m_n_malaeb/the-easy-guide-for-building-python-collaborative-filtering-recommendation-system-in-2017-d2736d2e92a8
# userID itemID rating timestamp
#from surprise import Reader, Dataset
## Define the format
#reader = Reader(line_format='user item rating timestamp', sep='\t')
## Load the data from the file using the reader format
#data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)
###################
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}

# n_epochs: The number of iteration of the ALS procedure. 
# lr_all – The learning rate for all parameters. Default is 0.005.
# Verbose = 2, this shows lots of useful output
# http://surprise.readthedocs.io/en/stable/evaluate.html
grid_search = GridSearch(SVD, param_grid, measures = ['RMSE'], verbose = 2)
# grid_search = GridSearch(SVD, param_grid, measures = ['MAE'], verbose = True)
# Perform the grid search
grid_search.evaluate(data)

# Print out the average RMSE for each fold and corresponding param pairs
rmse_result = grid_search.cv_results['RMSE']
param_pairs = grid_search.cv_results['params']

for i in range(len(rmse_result)):
    print("RMSE: {0}, Params: {1}".format(rmse_result[i], param_pairs[i]))
    
# Once our grid search is complete we can get the best model parameters by using the best_estimator attribute.
# Pickup best model from grid search
algo = grid_search.best_estimator['RMSE']

# Retrain on full set
trainset = data.build_full_trainset()
algo.train(trainset)

#userid = str(196)
#itemid = str(302)
#actual_rating = 4
#print algo.predict(userid, 302, 4)
#pred = algo.predict('userid', 'itemid',actual_rating)

# 1st: user id. 2nd: item id.
pred = algo.predict('374', '500')


print("Prediction Object:")
pred

pred = algo.predict('374', '500')

print("Prediction Object:")
pred


print("Predicted Rating:")
pred[3]

#########################
# http://surprise.readthedocs.io/en/stable/prediction_algorithms.html
# change the prediction algorithm to knn
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
# http://surprise.readthedocs.io/en/stable/similarities.html
#sim_options = {'name': 'pearson',
#               ''user_based': True
#               }

algo_1 = KNNBasic(sim_options= sim_options)
trainset = data.build_full_trainset()
algo_1.train(trainset)

pred = algo_1.predict('374', '500')

print("Prediction Object:")
pred

print("Predicted Rating:")
pred[3]


# print('Using ALS')
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
algo_2 = BaselineOnly(bsl_options=bsl_options)

trainset = data.build_full_trainset()
algo_2.train(trainset)

pred = algo_2.predict('374', '500')

print("Prediction Object:")
pred

print("Predicted Rating:")
pred[3]

#Predicting all missing entries
#First lets start by visualising our matrix of all observed entries.
#This matrix is quite sparse.
import numpy as np

n_users = trainset.n_users
n_items = trainset.n_items

M = np.zeros((n_users, n_items))

raw_ratings = data.raw_ratings

# rating= raw_ratings[1]
for rating in raw_ratings:
    r = int(rating[0]) - 1
    c = int(rating[1]) - 1
    v = rating[2]
    
    M[r, c] = v

import matplotlib.pyplot as plt 

fig = plt.figure()
plt.imshow(M)
fig

# Now with our trained SVD algorithm we can create a list of all user/item pairs in our dataset to fill in the rating matrix.

# What data did we end up with?

rows, cols = np.meshgrid(np.linspace(0, n_users-1, n_users), np.linspace(0, n_items-1, n_items), indexing = 'ij')

rows = np.ravel(rows)
cols = np.ravel(cols)

fullset = list()

for i in range(n_users * n_items):
    fullset.append( (str(int(rows[i])), str(int(cols[i])), 0) )
    
preds = algo.test(fullset)

X = np.zeros((n_users, n_items))
# e= preds[1]
# e[3]
vals = [e[3] for e in preds]

X[[int(r) for r in rows], [int(c) for c in cols]] = vals

fig = plt.figure()
plt.imshow(X)
fig

#Train/Test split in Surprise
#To perform a train/test split in Surprise you could do something like this:



data = Dataset.load_builtin('ml-100k')
raw_ratings = data.raw_ratings

random.shuffle(raw_ratings)

# A = 90% of the data, B = 10% of the data
threshold = int(.9 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = A_raw_ratings  # data is now the set A
data.split(n_folds=3)

# Then you can evaluate accuracy on the test set by
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)

predictions[0:9]


#####################################
#Low-Rank Matrix Completion (Optional)

import numpy as np
import pandas as pd
import scipy.sparse.linalg as sparse
import scipy as sp

def solve_nn(X, lam):
    # Solves the proximal nuclear norm problem
    # min lambda * |X|_* + 1/2*|X - Y|^2
    # solved by singular value shrinking
    
    # Cai, J. F., Candès, E. J., & Shen, Z. (2010).
    # A singular value thresholding algorithm for matrix completion.
    # SIAM Journal on Optimization, 20(4), 1956-1982.
    
    U, s, V = sp.linalg.svd(X, full_matrices=False)
    
    new_s = np.maximum(s - lam, 0)
    
    return U.dot(np.diag(new_s)).dot(V), new_s
    

def solve_lrmc(M, omega, tau = 1, mu = 0.1, rho =1 , iterations = 100, tol = 10e-6):
    # Solves the following problem
    # min_A    tau * | A |_* 
    # s.t. P.*M = P.*A + P.*E

    # which we convert to the unconstrained problem
    # min_A   tau * | A |_* + < Y, P.*A - P.*M > + mu/2 | P.*A - P.*M |_F^2

    # Adapted from
    # Lin, Z., Liu, R., & Su, Z. (2011).
    # Linearized alternating direction method with adaptive penalty for low-rank representation.
    # In Advances in neural information processing systems (pp. 612-620).
    
    f_vals = np.zeros(iterations)
    last_f_val = np.inf
    
    P = np.zeros(M.shape)
    P[omega] = 1
    
    Y = np.zeros(M.shape)
    A = np.zeros(M.shape)
    
    for k in range(iterations):
        partial = mu * (P*A - (P*M - 1/mu * Y))
        V = A - 1/rho * partial
        
        A, s = solve_nn(V, tau/rho)
        
        Y = Y + mu * (P*A - P*M)
        
        f_vals[k] = tau * np.sum(s)
        
        if (np.abs(f_vals[k] - last_f_val) <= tol):
            break
        else:
            last_f_val = f_vals[k]
        
    return A, f_vals

#Example
#Below we have a reference implementation and an example of usage on a synthetic dataset.
#First let's define a function to solve the LRMC and a helper function to solve the nuclear norm proximal problem.

# https://en.wikipedia.org/wiki/Matrix_completion#Low_rank_matrix_completion

# Then create some synthetic low-rank data
# 200 x 100  = 200 x 5 by 5 x 100
m = 200
n = 100
r = 5

A = np.dot(np.random.rand(m, r), np.random.rand(r, n))

print(np.linalg.matrix_rank(A))

fig = plt.figure()
plt.imshow(A)
fig


# Then randomly sample from the data.  MM  is our partially observed ratings matrix
sample_prop = 0.3

omega_linear = np.random.choice(int(m*n), int(m*n*sample_prop))

omega  = np.unravel_index(omega_linear, (m, n))

M = np.zeros(A.shape)
M[omega] = A[omega]

fig = plt.figure()
plt.imshow(M)
fig

# Finally solve the LRMC objective

print("Solving...")

X, fvals = solve_lrmc(M, omega)

print("Done")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
ax1.imshow(A)
ax1.set_title('Original')
ax2.imshow(M)
ax2.set_title("Observed")
ax3.imshow(X)
ax3.set_title("Recovered")
fig

#Compare the first row of A , M and X.
#Note that A and X are very close.
print(A[0,:])

print(M[0,:])

print(X[0,:])





