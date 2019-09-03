#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:17:16 2018

@author: Junbin Gao  All Copyright
"""

from sklearn.feature_extraction.text import TfidfTransformer


# Initiate the transformer
transformer = TfidfTransformer(smooth_idf=False)
# Check what it is
transformer  

# Corpus with three different terms and their counts in a corpus of 6 documents
counts = [[3, 0, 1],
           [2, 0, 0],
           [3, 0, 0],
           [4, 0, 0],
           [3, 2, 0],
           [3, 0, 2]]

# Transform the corpus
tfidf = transformer.fit_transform(counts)

# This is the transformed feature matrix for 6 documents
# This matrix can be pipelined into a machine learning algorithm
# Each row is normalized to have unit Euclidean norm:
X = tfidf.toarray()  


# If we are only given the entire corpus, rather than the counts, then we can 
# do the following way 
corpus = [
     'This is the first document.',
     'This is the second second document.',
     'And the third one.',
     'Is this the first document?',
]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(corpus)

# You can check all the terms 
vectorizer.vocabulary_
# Check the extracted features
X = X_tfidf.toarray()
# Please explain the meaning of 
X[0]

# A more complicated example
import pandas as pd
# This dataset from airbnb consists of texts hosters describe their hotels/rooms
# As there are possible non-English words, we use ISO-8859-1 encoding to read csv
descr = pd.read_csv('Room_Description.csv', encoding = "ISO-8859-1", engine='python')

"""
You cann set the number of features you wish to extract.  For larger corpus, this
number can be large, e.g., 3000
"""
n_features = 1000

"""
Let us define the tf-idf vecotorizer first. Here we set
max_df = 0.99 means ignoring terms that appear in more than 99% of the documents.
min_df = 3 means ignoring terms that appear in less than 2 documents.
stop_words = 'english' means we will use a built-in list of words to be removed 
    such as "a", "the" etc.
"""
tfidf_vectorizer = TfidfVectorizer(max_df = 0.99, min_df=3, max_features=n_features, stop_words='english')

tfidf_vectorizer.fit(descr['description']) 

X_tfidf = tfidf_vectorizer.transform(descr['description'])
X = pd.DataFrame(X_tfidf.toarray())
