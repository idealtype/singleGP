import warnings 
from copy import copy

import numpy as np
import numpy.random as npr
import GPy
from GPy.util.linalg import pdinv, symmetrify, dpotrs, dpotri
from GPy.util import choleskies

from GPtools.kernCompute import rbfKernCompute
from GPtools.kernGradient import rbfKernGradient
from collections import namedtuple


param = namedtuple('params_rbf', ['prec', 'var'])

class singleGP():
    '''
    A class for Gaussian Process model
    '''
    def __init__(self, X, y, kern_type='RBF'):

        self._X = X
        self._y = y
        self._nobs = X.shape[0]
        self._kern_type = kern_type

        # initalize
        if self._kern_type=='RBF':
            
            # initialize parameters 
            self._nparam = 2
            self._param = param(prec=1e3, var=1e3)
            self._sig2 = 1e3

            # initialize important matrices and quantities
            self.compute_mat(param=self._param, sig2=self._sig2)
            self._loglik = None

        else: 
            raise NotImplementedError


    def compute_mat(self, param, sig2):
        
        '''
        compute important matrices
        '''
        # prec = param.prec
        # var = param.var

        # _K = K + (sig**2)*I
        self._K = rbfKernCompute(param, self._X)['K'] + sig2*np.eye(self._nobs)
        
        self._Kinv, self.L, self.Linv, self.logDet_K = pdinv(self._K , 10)
        self.a = self._Kinv @ self._y


    def nll(self, H, **kwargs):
        
        '''
        return the negative log ikelihood of the GP model.

        - params
        - sig
        '''

        # update vector and matrices with given parameters
        self.model_update(H)

        # calculate log-likelihood
        self._loglik = -.5 * (
            self._y.T @ self.a 
            + self.logDet_K
            + self._nobs * np.log(2*np.pi)
        ) 

        return -self._loglik.item()


    def grad_nll(self, H, **kwargs):

        '''
        return gradient of the negative log likelihood.

        - params
        - sig
        '''

        # update vector and matrices with given parameters
        self.model_update(H)

        A = .5*(self.a @ self.a.T - self._Kinv)
        dLd_K_Param = rbfKernGradient(
            H=self._param, partialMat=A, x=self._X
        ) 

        factor_prec = self._param.prec
        factor_var = self._param.var
        factor_sig2 = self._sig2

        dLdprec = dLd_K_Param['matGradPreU']*factor_prec
        dLdvar = dLd_K_Param['matGradSens']*factor_var
        dLdsig2 = np.trace(A)*factor_sig2

        grad = np.array([dLdprec, dLdvar, dLdsig2])

        return -grad
    
    def predict(self, Xpred, H=None):

        '''
        Derive a predictive distribution at Xpred
        '''

        if H is not None: self.model_update(H)
        
        kstar = rbfKernCompute(H=self._param, x=self._X, x2=Xpred)['K']
        kstarstar = rbfKernCompute(H=self._param, x=Xpred)['K']
        
        self.E_fstar = kstar.T @ self.a 
        self.V_fstar = kstarstar - kstar.T @ self._Kinv @ kstar
        self.V_ystar = self.V_fstar + self._sig2*np.eye(Xpred.shape[0])

        return self.E_fstar, self.V_fstar, self.V_ystar


    def model_update(self, H):

        """
        Update the GP model with the new parameter H 

        - H: list, parameters
        """
        
        # Update parameters with given values H
        # If nonnegative, transform it using the exponetial function 
        if self._kern_type == 'RBF':
            self._param = self._param._replace(
                prec=GPy.util.functions.clip_exp(H[0]),
                var=GPy.util.functions.clip_exp(H[1])
            )
            self._sig2 = GPy.util.functions.clip_exp(H[2])  
             
        else: 
            raise NotImplementedError

        # update matrices
        self.compute_mat(param=self._param, sig2=self._sig2)

        return
    

