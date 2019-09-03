# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:10:39 2018

@author: jgao5111
"""

from sklearn.feature_extraction.text import CountVectorizer

"""
This example uses a very small corpus to show you how extract the count features
of terms in the corpus
"""

# First let us prepare the model
vectorizer = CountVectorizer()

# Define the corpus which is a list of statements, each of which can be regarded 
# as a document. Here we have 4 documents
corpus = [
      'This is the first document.',
      'This is the second second document.',
      'And the third one.',
      'Is this the first document?']

# We fit our model to this corpus, and the fitted model contains all the learned
# information about the corpus
vectorizer.fit(corpus)

#If you wish to see what words' counts are collected, check 
vectorizer.get_feature_names()

# Now use the trained model to transform the entire corpus into the featrure of counts 
# of terms
X = vectorizer.transform(corpus)

# Or you can do the above in one step
X = vectorizer.fit_transform(corpus)
# Let us show this set of extracted count features
X.toarray() 


# With the trained model, you can transform a new document into a feature 
X_new = vectorizer.transform(["This is a text document to analyze."])
# Before you print out the vector X_new, can you guess what it is?


 