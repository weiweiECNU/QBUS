import numpy as np
from scipy.signal import lfilter
import statsmodels.api as sm

# True model parameters for an AR(1)
nobs = int(1e3)
true_phi = 0.5
true_sigma = 1**0.5

# Simulate a time series
np.random.seed(1234)
disturbances = np.random.normal(0, true_sigma, size=(nobs,))
endog = lfilter([1], np.r_[1, -true_phi], disturbances)

# Construct the model for an ARMA(1,1)
class ARMA11(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(ARMA11, self).__init__(endog, k_states=2, k_posdef=1,
                                     initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1., 0]
        self['transition'] = [[0, 0],
                              [1., 0]]
        self['selection', 0, 0] = 1.

    # Describe how parameters enter the model
    def update(self, params, transformed=True, **kwargs):
        params = super(ARMA11, self).update(params, transformed, **kwargs)

        self['design', 0, 1] = params[0]
        self['transition', 0, 0] = params[1]
        self['state_cov', 0, 0] = params[2]

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0.,0.,1]  # these are very simple

# Create and fit the model
mod = ARMA11(endog)
res = mod.fit()
print(res.summary())

#%%
import statsmodels.api as sm
from pandas_datareader.data import DataReader

gdp = DataReader('GDPC1', 'fred', start='1959', end='12-31-2014')

# Create the model, here an SARIMA(1,1,1) x (0,1,1,4) model
mod = sm.tsa.SARIMAX(gdp, order=(1,1,1), seasonal_order=(0,1,1,4))

# Fit the model via maximum likelihood
res = mod.fit()
print(res.summary())