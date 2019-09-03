#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:28:55 2018

@author: steve
"""

import pandas as pd

# Read a CSV file, this returns a DataFrame object
# A DataFrame is similar to a spreadsheet/table
happiness_df = pd.read_csv('happiness_2016.csv')

# A single column of a DataFrame is a Series
country_series = happiness_df['Country']

# You can index into Dataframes and Series lots of ways
# Multiple columns
sub_dataframe = happiness_df[ ['Country', 'Region'] ]

# By column number
region_series = happiness_df.iloc[:, 1]

# By row number
row_10 = happiness_df.iloc[9, :]

# Single Cell
happy_score_sweden = happiness_df.iloc[9, 3]

# We can query a dataframe very easily
# Find all countries (rows) with a happiness score > 7.4
top_scorers = happiness_df[ happiness_df['Happiness Score'] > 7.4 ]

# Find all countries in SEA with Freedom > .5
free_sea = happiness_df[ (happiness_df['Region'] == 'Southeastern Asia') & (happiness_df['Freedom'] > 0.5 ) ]

# Check if there are any missing or corrupt values
print("Contains missing values") if happiness_df.isnull().values.any() else print("No missing values")

# Delete rows containing NaN values
happiness_df.dropna(inplace=True)

# Or you can replace them with a placeholder value
happiness_df.fillna(value="PLACEHOLDER", inplace=True)