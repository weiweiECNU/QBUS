# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#train = pd.read_csv("review_train.csv")
#test = pd.read_csv("review_test.csv")

train = pd.read_csv("review_train.csv")
test = pd.read_csv("review_test.csv")

train.drop(["PRODUCT_ID"],axis = 1,inplace = True)
test.drop(["PRODUCT_ID"],axis = 1,inplace = True)



