# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:50:42 2018

@author: Jiahao
"""

import numpy as np
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
 
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
        self.ssm['design'] = np.array([1])     #for Z_t
        self.ssm['transition'] = np.array([1]) #for T_t
        self.ssm['selection'] = np.array([1])  #for R_t
  
    # We give names to parameters sigma^2_e, sigma^2_eta and sigma^2_xi
    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level']
     
    # initial value for the parameters
    @property
    def start_params(self):
        return [np.std(self.endog)]*2
     
    # Because our parameters are variances which are greater than zero (a condition)
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

#%%
# We load the time series data
visitors = pd.read_csv('AustralianVisitors.csv')

y = visitors['No of Visitors']
y.index = pd.date_range('01/01/1991', '31/12/2016', freq='MS')

#%%
# Setup the model
model_1 = LocalLevel(y)
#%%
# Fit it using MLE (recall that we are fitting the two variance parameters)
res = model_1.fit(disp=False)
print(res.summary())

#%%
#Finally, we can do post-estimation prediction and forecasting.
predict = res.get_prediction()
forecast = res.get_forecast('2018')

# Plot the results
fig, ax = plt.subplots(figsize=(10,4))
y.plot(ax=ax, style='k.', label='Observations')
predict.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')

forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')

legend = ax.legend(loc='lower left')

#%%

"""
The next example shows how to define
   Univariate Holt's Linear Trend Model 
     
"""
class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    # statsmodel defines the basic MLEModel with all the functionalities we need
    def __init__(self, endog):
        # The dimension of the states, this is 2 because states are
        # alpha_t and nu_t, see the slide 
        k_states = 2
         
        # int – The dimension of a guaranteed positive definite covariance matrix describing the shocks in the measurement equation.
        k_posdef = 2
 
        # Initialize the state space
        # Note endog is the argument for the time series data
        # we will use approximate-diffusing method to initialize u_{1|0} and P_{1|0}, see the slide       
        super(LocalLinearTrend, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )
 
        # Initialize the matrices
        self.ssm['design'] = np.array([1, 0])    #for Z_t
        self.ssm['transition'] = np.array([[1, 1],
                                       [0, 1]])  #for T_t
        self.ssm['selection'] = np.eye(k_states) #for R_t
 
        # Cache some indices, for state cov Q_t the parameters are at diagonal
        self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)
 
    # We give names to parameters sigma^2_e, sigma^2_eta and sigma^2_zeta
    @property
    def param_names(self):
        return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']
     
    # initial value for the parameters
    @property
    def start_params(self):
        return [np.std(self.endog)]*3
     
    # Because our parameters are variances which are greater than zero (a condition)
    # these transforms are used to transform the parameters to an unconditional 
    # space for unconstrained optimisation
    def transform_params(self, unconstrained):
        return unconstrained**2
 
    def untransform_params(self, constrained):
        return constrained**0.5
     
    # We must define this functions
    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
         
        # Observation covariance  i.e., H_t
        self.ssm['obs_cov',0,0] = params[0]
 
        # State covariance   i.e., Q_t
        self.ssm[self._state_cov_idx] = params[1:]

    
#%%
# We load the time series data
df2 = pd.read_csv('OxCodeAll.csv')  
# Preparing index for visualisation
df2.index = pd.date_range(start='%d-01-01' % df2.date[0], end='%d-01-01' % df2.iloc[-1, 0], freq='AS')
 
# Log transform
df2['lff'] = np.log(df2['ff'])
 
# Setup the model
model2 = LocalLinearTrend(df2['lff'])
 
# Fit it using MLE (recall that we are fitting the three variance parameters)
res2 = model2.fit(disp=False)
print(res2.summary())

#%%
predict = res2.get_prediction()
forecast = res2.get_forecast('2014')

fig, ax = plt.subplots(figsize=(10,4))
 
# Plot the results
df2['lff'].plot(ax=ax, style='k.', label='Observations')
predict.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')
predict_ci = predict.conf_int(alpha=0.05)
predict_index = np.arange(len(predict_ci))
ax.fill_between(predict_index[2:], predict_ci.iloc[2:, 0], predict_ci.iloc[2:, 1], alpha=0.1)
 
forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')
forecast_ci = forecast.conf_int()
forecast_index = np.arange(len(predict_ci), len(predict_ci) + len(forecast_ci))
ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.1)
 
# Cleanup the image
ax.set_ylim((4, 8));
legend = ax.legend(loc='lower left');
