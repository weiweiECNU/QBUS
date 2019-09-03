import pandas as pd
data1 = pd.read_csv("parse_ex1.csv", parse_dates= ["Date"],index_col= 'Date' )
data2 = pd.read_csv("parse_ex2.csv", parse_dates=["Time"],index_col="Time")
print(data2)