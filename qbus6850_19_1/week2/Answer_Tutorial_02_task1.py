#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:28:55 2018

@author: steve
"""

import numpy as np

# Create two vectors
a = np.array([1, 2, 0])
b = np.matrix([2, -1, 10]).transpose()

# Inner product of two vectors
c = np.dot(a, b)
print(c)

# Are vectors a and b orthogonal?
print("Orthogonal") if c == 0 else print("Not Orthogonal")

# Create a matrix
A = np.matrix([ [1, -1, 2], [0, -3, 1] ])

# Multiply matrix and a vector
A_b = np.dot(A, b)
print(A_b)

# What is the size of the result?
print(A_b.shape)

# Multiply matrix and its transpose
At_A = np.dot(A.transpose(), A)
print(At_A)

# What is the size of the result?
print(At_A.shape)