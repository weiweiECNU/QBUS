#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 00:01:46 2019

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect("housing.db")
house = pd.read_sql("SELECT * FROM houses;", conn)
#house['Alley'] = house['Alley'].apply(lambda x: "None" if pd.isnull(x) else x)
#house['Bsmt Qual'] = house['Bsmt Qual'].apply(lambda x: "None" if pd.isnull(x) else x)
#house['Bsmt Cond'] = house['Bsmt Cond'].apply(lambda x: "None" if pd.isnull(x) else x)
#house['Bsmt Exposure'] = house['Bsmt Cond'].apply(lambda x: "None" if pd.isnull(x) else x)
#house['BsmtFin Type 1'] = house['BsmtFin Type 1'].apply(lambda x: "None" if pd.isnull(x) else x)
#house["BsmtFin Type 2"] = house["BsmtFin Type 2"].apply(lambda x: "None" if pd.isnull(x) else x)
#
#house["Fireplace Qu"] = house["Fireplace Qu"].apply(lambda x: "None" if pd.isnull(x) else x)
#house["Garage Type"] = house["Garage Type"].apply(lambda x: "None" if pd.isnull(x) else x)
#house["Garage Finish"] = house["Garage Finish"].apply(lambda x: "None" if pd.isnull(x) else x)
#house["Garage Qual"] = house["Garage Qual"].apply(lambda x: "None" if pd.isnull(x) else x)
#house["Garage Cond"] = house["Garage Cond"].apply(lambda x: "None" if pd.isnull(x) else x)
#house["Pool QC"] = house["Pool QC"].apply(lambda x: "None" if pd.isnull(x) else x)
#house["Fence"] = house["Fence"].apply(lambda x: "None" if pd.isnull(x) else x)
#house["Misc Feature"] = house["Misc Feature"].apply(lambda x: "None" if pd.isnull(x) else x)
