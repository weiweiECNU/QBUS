#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:28:55 2018

@author: steve
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read a CSV file, this returns a DataFrame object
# A DataFrame is similar to a spreadsheet/table
happiness_df = pd.read_csv('happiness_2016.csv')

# Find all countries in SEA with Freedom > .5
free_sea = happiness_df[ (happiness_df['Region'] == 'Southeastern Asia') & (happiness_df['Freedom'] > 0.5 ) ]

# Pyplot draws each plot on a Figure
# Think of a Figure as a blank canvas
# By default Pyplot will draw to the last Figure you created
plt.figure()

# Bar Plot
# Plot the happiness score of SEA countries with Freedom > .5
ind = np.arange(len(free_sea))
plt.bar(ind, free_sea['Happiness Score'])

plt.xticks(ind, free_sea['Country'])

plt.xlabel("Country")

plt.ylabel("Happiness Score")

plt.title("Happiness Score of SEA Countries with Freedom > 0.5")

# Line Plot
# Plot the Happiness Rank vs Life Expectancy
# Note that countries are already sorted by rank

plt.figure()

# Line plot with a label and custom colour
# Other optional parameters include linestyle and markerstyle
plt.plot(happiness_df['Health (Life Expectancy)'], label="Life Expectancy", color="red")

plt.xlabel("Happiness Rank")

plt.ylabel("Life Expectancy")

plt.title("Happiness Rank vs Life Expectancy")

# Activate the legend using label information
plt.legend()

