# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

comments = pd.read_csv("User_Comments.csv")[["CONTENT","CLASS"]]

from sklearn.model_selection import train_test_split
X = comments["CONTENT"]
y = comments["CLASS"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.feature_extraction.text import CountVectorizer

corpus = X_train.tolist()
vectorizer = TfidfVectorizer()
count = vectorizer.fit_transform(corpus)


