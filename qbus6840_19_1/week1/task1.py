# -*- coding: utf-8 -*-
import pandas as pd

a = pd.Series(['david', 'jess', 'mark', 'laura'])
print(a)
for i in a:
    if i != "mark":
        print(i)
