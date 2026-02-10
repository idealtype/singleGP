import numpy as np
import numpy.random as npr
from GPtools.kernCompute import rbfKernCompute
from collections import namedtuple

data = namedtuple(
    'bdata', ['X', 'y', 'ytrue']
)

class data_generator():
    
    """
    Synthetic data generator
    """
    
    def __init__(self, X, func='Gaussian'):
        self._X = X
        self._func = func
        self._nobs = X.shape[0]
    
    def generate(self, params, sig=0.1):
        
        if self._func == 'poly':
            ytrue = self._poly(self._X, params)
        elif self._func == 'Gaussian':
            ytrue = self._gaussian(self._X, params)
        else:
            raise NotImplementedError()
    
        y = ytrue + sig*npr.normal(size=self._nobs)[:, None]

        return data(X=self._X, y=y, ytrue=ytrue)
    
    def _poly(self, X, params):

        return params[0]*(X**2) + params[1]*X + params[3]
    
    def _gaussian(self, X, params):

        '''
        - params : namedtuple(prec, var)
        
        '''

        kern = rbfKernCompute(H=params, x=X)
        K = kern['K']
        
        return npr.multivariate_normal(mean=np.zeros(self._nobs), cov=K)[:,None]
        


