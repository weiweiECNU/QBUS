#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 08:34:51 2018

@author: jbgao
adopted from the example in the article
http://www.chadfulton.com/files/fulton_statsmodels_2017_v1.pdf
"""

import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt


"""
We will use statsmodels package to do state-space modeling.
statsmodels offers functionalities for us to do prediction, filtering and 
smoothing for state-space models
Some models are fully implemented in statsmodel such as ARIMA models
If we have a model which is not implemented in statsmodel, we shall 
follow a standard procedure to define the model ourself.

This example shows how to define

    Exponential Model 
    
"""
class LocalLevel(sm.tsa.statespace.MLEModel):
    # statsmodel defines the basic MLEModel with all the functionalities we need
    def __init__(self, endog):
        # The dimension of the states, this is 1 because state is
        # mu_t, see the slide 
        k_states = 1
        
        # The dimension of the noise terms, i.e., xi_t
        k_posdef = 1

        # Initialize the statespace
        # Note endog is the argument for the time series data
        # we will use approximate-diffusing method to initialize u_{1|0} and P_{1|0}, see the slide       
        super(LocalLevel, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=10
        )

        # Initialize the matrices
        self.ssm['design'] = np.array([1])       # for Z_t
        self.ssm['transition'] = np.array([1])   # for T_t
        self.ssm['selection'] = np.array([1])    # for R_t


    # We give names to parameters sigma^2_e, sigma^2_eta and sigma^2_xi
    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level']
    
    # initial value for the parameters
    @property
    def start_params(self):
        return [np.std(self.endog)]*2
    
    # Because our parameters are varances which are greater than zero (a condition)
    # these transforms are used to transform the parameters to an unconditional 
    # space for unconstrained optimisation
    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5
    
    # We must define this functions
    def update(self, params, *args, **kwargs):
        params = super(LocalLevel, self).update(params, *args, **kwargs)
        
        # Observation covariance  i.e., H_t
        self.ssm['obs_cov',0,0] = params[0]

        # State covariance   i.e., Q_t
        self.ssm['state_cov', 0, 0] = params[1]


# We load the time series data
nile = sm.datasets.nile.load_pandas().data['volume']
nile.index = pd.date_range('1871', '1970', freq='AS')

# Setup the model
nile_model_1 = LocalLevel(nile)

## Compute the loglikelihood at values specific to the nile model
#print(nile_model_1.loglike([15099.0, 1469.1]))
#
## Try computing the loglikelihood with a different set of values; notice that it is different
#print(nile_model_1.loglike([10000.0, 1.0]))
# 
#
## Retrieve filtering output
#nile_filtered_1 = nile_model_1.filter([15099.0, 1469.1])
## print the filtered estimate of the unobserved level
#print(nile_filtered_1.filtered_state[0])
## [ 1103.34065938  ... 798.37029261 ]
#print(nile_filtered_1.filtered_state_cov[0, 0])
## [ 14874.41126432  ... 4032.15794181 


# Fit it using MLE (recall that we are fitting the two variance parameters)
res = nile_model_1.fit(disp=False)
print(res.summary())


#Finally, we can do post-estimation prediction and forecasting.
# Perform prediction and forecasting
predict = res.get_prediction()
forecast = res.get_forecast('1997')

fig, ax = plt.subplots(figsize=(10,4))

# Plot the results
nile.plot(ax=ax, style='k.', label='Observations')
predict.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')
 
forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')
 
# Cleanup the image
#ax.set_ylim((4, 8));
legend = ax.legend(loc='lower left');      