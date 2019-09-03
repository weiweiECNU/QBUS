
"""
Created on Fri Oct 13 15:39:21 2017

@author: cwan6954
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from keras.layers.core import Dense
from keras.models import Sequential

np.random.seed(0)

df = pd.read_csv("Advertising.csv", index_col=0)

df.head()

# https://keras.io/

# Usually with Neural Networks we scale all features to be in the range (0-1) to reduce the effect of feature domination and gradient vanishing. 
# This particularly helps for deep networks. So it is a good habit to get into.
# Don't worry we can easily transform the data back into the original range later with the scaler object.

# http://scikit-learn.org/stable/modules/preprocessing.html
# The transformation is given by:
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

scaler = MinMaxScaler(feature_range=(0, 1))

data = scaler.fit_transform(df.values)

df_scaled = pd.DataFrame(data, columns=df.columns)

df_scaled.head()

# Now we can do a train/test split
y = df_scaled["Sales"]
X = df_scaled[ df_scaled.columns.difference(["Sales"]) ]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# There are three stages to building neural networks with Keras:
# defining
# compiling
# training

# Define the network architecture
model = Sequential()

n_features = X_train.shape[1]
# https://keras.io/layers/core/
# First layer, input size is the number of input features and output size is 1
# We are doing a linear regression so our activation should be linear
# model.add(Dense(1, input_dim=n_features, activation='sigmoid', use_bias=True))
model.add(Dense(1, input_dim=n_features, activation='linear', use_bias=True))

# Use mean squared error for the loss metric and use the ADAM backprop algorithm
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the network (learn the weights)
# We need to convert from DataFrame to NumpyArray
# next week we will learn more about batch gradient descent and stochastic gradient descent
history = model.fit(X_train.values, y_train.values,  epochs=100, batch_size=1, verbose=2, validation_split=0)
# abc= X_train.values
# Visualising a Network
# Often our network can get quite complicated and a visualisation can help with understanding what is happening at each layer.
# Below is an example of how to plot a model as a Figure

# https://pypi.python.org/pypi/pydot
# from keras.utils.vis_utils import model_to_dot
# from IPython.display import Image, display
#
# dot_obj = model_to_dot(model, show_shapes = True, show_layer_names = True)
#
# display(Image(dot_obj.create_png()))


#You can also save your figure to an PDF or image file
from keras.utils import plot_model
#
# plot_model(model, to_file='model.pdf', show_shapes = True, show_layer_names = True)
# summary
model.summary()

#To view the weights for every layer you can use a loop. 
#Since we only have a single layer we only have one array of weights plus the bias term.

print(len(model.layers))
# last one is the bias unit
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

# generate prediction
y_pred = model.predict(X_train.values)

import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.plot(y_train.values,'r')

## transform back
#
#df_raw= np.column_stack((X_train.values, y_train.values))
#raw_inverse = scaler.inverse_transform(df_raw)
#
## column stack
# df_predict= np.column_stack((X_train.values,y_pred))
# pred_inverse = scaler.inverse_transform(df_scaled)

# Comparison to OLS
# Now lets compare with the parameters estimated by sklearn's OLS function. The parameters are very close!

import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

ols = LinearRegression(fit_intercept=True)
ols.fit(X_train, y_train)

ols_coefs = np.concatenate( ([ols.intercept_], ols.coef_), axis = 0 )
print("OLS Coefs: {0}".format(ols_coefs))

weights = model.layers[0].get_weights()

nn_coefs = np.concatenate( (weights[1], weights[0][:,0]) , axis = 0 )
print("NN Coefs: {0}".format(nn_coefs))

# Binary Classification Network
# Use an activation function on output layer to clamp output to binary
# Softmax probability distribution over the outcomes.

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout


np.random.seed(0)

bank_df = pd.read_csv("bank.csv")

X = bank_df.iloc[:, 0:-1]
y = bank_df['y_yes']

scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled = pd.DataFrame(scaler.fit_transform(X.values), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

n_features = X_train.shape[1]

model = Sequential()

model.add(Dense(64, input_dim=n_features, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#model.add(Dense(64, input_dim=n_features, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))


# Use mean squared error for the loss metric and use the ADAM backprop algorithm
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the network (learn the weights)
# We need to convert from DataFrame to NumpyArray
history = model.fit(X_train.values, y_train.values,  epochs=2, batch_size=16, verbose=1, validation_split=0, class_weight={0:0.0001, 1:100})
y_pred = model.predict(X_test.values)

print(classification_report(y_test, y_pred.astype(int)))
print(confusion_matrix(y_test, y_pred.astype(int)))

from sklearn.utils import class_weight
# https://keras.io/layers/about-keras-layers/
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)   

# some summary stats
from keras.utils.vis_utils import model_to_dot
from IPython.display import Image, display

dot_obj = model_to_dot(model, show_shapes = True, show_layer_names = True)

display(Image(dot_obj.create_png()))
plot_model(model, to_file='model_classification_1.pdf', show_shapes = True, show_layer_names = True)

model.summary()

#To view the weights for every layer you can use a loop. 
#Since we only have a single layer we only have one array of weights plus the bias term.

print(len(model.layers))
# last one is the bias unit
for layer in model.layers:
    weights = layer.get_weights()
    print(weights)

#############################

# Multi-Class Classification Network
from keras.optimizers import SGD
np.random.seed(0)

wine_df = pd.read_csv('winequality-white.csv', delimiter=";")

X = wine_df.iloc[:, :-1]
y = wine_df.iloc[:, -1]

y = pd.get_dummies(y)

scaler = MinMaxScaler(feature_range=(0, 1))

X_scaled = pd.DataFrame(scaler.fit_transform(X.values), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=1)

n_features = X_train.shape[1]

# create model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=n_features))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train.values, y_train.values, epochs=1, batch_size=16)

y_pred = model.predict(X_test.values)
abc= y_test.values
y_te = np.argmax(y_test.values, axis = 1)
y_pr = np.argmax(y_pred, axis = 1)

print(np.unique(y_pr))
print(np.unique(y_te))

print(classification_report(y_te, y_pr))

print(confusion_matrix(y_te, y_pr))


#Method 1 - Keras Wrappers
#We can use cross_val_score through the use Keras' wrapper function KerasClassifier().

from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(0)

def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=n_features))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

sk_model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=16, verbose=0)

print("Running...")
cv_scores = cross_val_score(sk_model, X_train.values, y_train.values, scoring = 'neg_log_loss')

print(cv_scores)


# Method 2 - Manual K-Folds¶

from sklearn.model_selection import KFold

np.random.seed(0)

n_features = X_train.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=n_features))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

kf = KFold(3, shuffle=False)

cv_scores = list()

print("Running...")
for train_index, val_index in kf.split(X_train.T):
    X_cvtrain, X_cvval = X_train.iloc[train_index, :], X_train.iloc[val_index, :]
    y_cvtrain, y_cvval = y_train.iloc[train_index], y_train.iloc[val_index]

    model.fit(X_cvtrain.values, y_cvtrain.values, epochs=20, batch_size=16, verbose=0)
    
    score = model.evaluate(X_cvval.values, y_cvval.values, verbose=0)
    
    # First item in scores is the loss. We specified the loss as crossentropy so we take the negative of it.
    
    cv_scores.append(-score)
    # cv_score higher the better
print(cv_scores)



#Cross Validation - Hyper Parameter Optimisation
#Method 1 - GridSearchCV

from sklearn.model_selection import GridSearchCV

def create_model(optimizer='rmsprop'):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=n_features))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    # accurcy higher the better
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)

optimizers = ['rmsprop']
epochs = [5, 10, 15]
batches = [128]


param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, verbose=['2'])
grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid.fit(X_train.values, y_train.values)


# Cross Validation - Optimising Layers
# You can also combine GridSearchCV and the method below to optimise the hyperparameters and layers at the same time

total_layers = [  
            Dense(64, activation='relu', input_dim=n_features),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5)
         ]

final_layer = Dense(7, activation='softmax')

class model_builder:
    
    def __init__(self, layers, final_layer):
        self.layers = layers
        self.final_layer = final_layer
        
    def __call__(self):
        self.model = Sequential()
        
        for layer in self.layers:
            self.model.add(layer)

        self.model.add(self.final_layer)

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return self.model

    
cv_mean_scores = list()
print("Running...")
# 1,2,3
for i in range(1, len(total_layers)):
    
    layers = total_layers[:i]
    
    mdl_builder = model_builder(layers, final_layer)
    
    sk_model = KerasClassifier(build_fn=mdl_builder, epochs=20, batch_size=16, verbose=1)
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html    
    cv_scores = cross_val_score(sk_model, X_train.values, y_train.values, scoring = 'neg_log_loss')
       
    cv_mean_scores.append(np.mean(cv_scores))

# Put a new line in and then print out mean cv scores
# Each score is the CV score for a network with different layers
print("\n")
# higher the better, as negative log loss
print(cv_mean_scores)


print(len(mdl_builder.layers))

# the log loss is
# -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

# one epoch = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass. 
# The higher the batch size, the more memory space you'll need.
# number of iterations = number of passes, each pass using [batch size] number of examples. 
# To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).


























