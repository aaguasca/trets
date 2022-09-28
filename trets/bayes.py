#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE

import numpy as np
from scipy.stats import norm
import math
from math import factorial 
from decimal import Decimal
import warnings
from scipy.special import loggamma

__all__=[
    "BayesianProbability"
]

class BayesianProbability:
    """
    Bayesian probability applied to ON/OFF observations to obtain the posterior probability that given
    an observation with n_on events in the ON region, n_off events in the off region and a ratio
    between the integrated acceptance of the telescope in the ON/OFF regions, it can be produced
    for a source with an expected signal (mu_s).
    
    The results for high values of n_on and n_off may not be reliable. Everything is fine at least if 
    n_on < 110 and n_off<110, when an exact solution is given. Afterwards, who knows :). Thus, please,
    use this method with few events.
    
    The probability that there is a source in the ON/OFF observations is also included.
    """
    
    def __init__(self,n_on,n_off,alpha):
        self.n_on=n_on
        self.n_off=n_off
        self.alpha=alpha
        
    def _factorial(self,number):
        """
        Compute the factorial of number.
        
        Parameters
        ----------
        number: int
            Number to compute the factorial of it.
            
        Returns
        -------
        fact: int
            Factiorial value of number.
            
        """

        fact=factorial(number)

        return fact

    def poisson_dist(self,_lambda,n):
        """
        Probability mass function of a Poissonian distribution.
        
        Parameters
        ----------
        _lambda: float
            Expected value
        n: int
            Number of occurances.
            
        Returns
        -------
        Probability mass function given _lambda and n.
        
        """
        return ((_lambda**n)/(self._factorial(n)))*math.exp(-_lambda)
        
        
    def _normalized_coeff_binomial_expansion(self,event_i):
        """
        Probability that n - {i} events are background events.
                
        Parameters
        ----------
        event_i : int 
            i-th considered event.
           
        Returns
        -------
        C_i : int
            The normalized coefficient for the i-th ON event. i.e. the the probability for the
            n - {i} events are background events.
        
        """
        def eq_denominator():
            """
            Compute the denominator of the expansion of the normalized binomial expansion. i.e.
            the normalization factor necessary in order to have \sum{C_i}_{i}^{n_on}=1. Then,
            
            Returns
            -------
            denominator: int or float
                Value of the normalization factor.
            
            """
            event_j=np.arange(self.n_on+1, dtype=int)
            
            denominator=0   
            if self.n_on>110 or self.n_off>110:
                alpha_term=round(1+self.alpha**(-1))
                for j in event_j:
                    first_term=((alpha_term)**int(j))
                    second_term=(self._factorial(self.n_on+self.n_off-j)//self._factorial(self.n_on-j))            
                    denominator+=first_term*second_term
            else:
                alpha_term=1+self.alpha**(-1)
                for j in event_j:
                    first_term=((alpha_term)**int(j))
                    second_term=(self._factorial(self.n_on+self.n_off-j)/self._factorial(self.n_on-j))            
                    denominator+=first_term*second_term
                
            return denominator
              

        #compute denominator
        denominator=eq_denominator()
    
        #compute numerator
        if self.n_on>110 or self.n_off>110:
            warnings.warn("Warning: The value may not be precise! We are dealing with huge values")
    
            alpha_term=round(1+self.alpha**(-1))
            first_term=((alpha_term)**event_i)
            second_term=(self._factorial(self.n_on+self.n_off-event_i)//self._factorial(self.n_on-event_i)) 
            numerator=first_term*second_term
            
            C=numerator/denominator
            
        else:
            alpha_term=1+self.alpha**(-1)
            first_term=((alpha_term)**event_i)
            second_term=(self._factorial(self.n_on+self.n_off-event_i)/self._factorial(self.n_on-event_i)) 
            numerator=first_term*second_term
            
            C=numerator/denominator
        
        return C
    
    
    
    def _log_normalized_0_coeff_binomial_expansion(self):
        """
        Log Probability that n events are background events.
           
        Returns
        -------
        C_0 : int
            The normalized coefficient for the 0-th ON event. i.e. the the probability for the
            n events to be background events.
        
        """
        def eq_denominator():
            """
            Compute the denominator of the expansion of the normalized binomial expansion. i.e.
            the normalization factor necessary in order to have \sum{C_i}_{i}^{n_on}=1. Then,
            
            Returns
            -------
            denominator: int or float
                Value of the normalization factor.
            
            """
            event_j=np.arange(self.n_on+1, dtype=int)
            
            denominator=0   
            if self.n_on>110 or self.n_off>110:
                alpha_term=round(1+self.alpha**(-1))
                for j in event_j:
                    first_term=((alpha_term)**int(j))
                    second_term=(self._factorial(self.n_on+self.n_off-j)//self._factorial(self.n_on-j))            
                    denominator+=first_term*second_term
            else:
                alpha_term=1+self.alpha**(-1)
                for j in event_j:
                    first_term=((alpha_term)**int(j))
                    second_term=(self._factorial(self.n_on+self.n_off-j)/self._factorial(self.n_on-j))            
                    denominator+=first_term*second_term
                
            return denominator
              

        #compute denominator
        denominator=math.log(eq_denominator())
    
        #compute numerator
        if self.n_on>110 or self.n_off>110:
            warnings.warn("Warning: The value may not be precise! We are dealing with huge values")
    
            first_term = loggamma(self.n_on+self.n_off+1)
            second_term = loggamma(self.n_on+1)
            numerator=first_term-second_term
            
            C=numerator-denominator
            
        else:
            first_term=loggamma(self.n_on+self.n_off+1)
            second_term=loggamma(self.n_on+1) 
            numerator=first_term-second_term
            
            C=numerator-denominator
        
        return C    
    

    def posterior_proba(self,mu_s):
        """
        Compute the posterior probability for a source with an expected signal of mu_s.
        
        Parameters
        ----------
        mu_s: float
            Expected signal of the source.
            
        Returns
        -------
        proba: float
            Posterior probability
        
        """
        event_i=np.arange(self.n_on+1, dtype=int)
        self.proba=[]
        if self.n_on>110 or self.n_off>110:
            if mu_s==0:
                C=self._log_normalized_0_coeff_binomial_expansion()                
                self.proba=np.exp(np.array(C))
            else:
                for i in event_i:
                    C=self._normalized_coeff_binomial_expansion(int(i))#assert i and mu_s are int type
                    poss=self.poisson_dist(mu_s,int(i))#assert i and mu_s are int type
                    self.proba.append(C*poss)
        else:
            if mu_s==0:
#                 C=self._normalized_coeff_binomial_expansion(event_i=0) 
                C=self._log_normalized_0_coeff_binomial_expansion()
                self.proba=np.exp(np.array(C))
            else:
                for i in event_i:
                    C=self._normalized_coeff_binomial_expansion(int(i))#assert i and mu_s are int type
                    poss=self.poisson_dist(mu_s,int(i))#assert i and mu_s are int type
                    self.proba.append(C*poss)

        self.proba=np.array(self.proba)
        return self.proba.sum()
    
    def detection_significance(self):
        """
        Compute the detection significance. i.e. the how confidence we are 
        that the data departs from an only background data (mu_s=0)
        
        Parameters
        ----------
        
        Returns
        -------
        significance: float
            Significance that there is only background in the data.
        """
        detection_proba = self.posterior_proba(0)
        significance = self.proba_to_sigma(detection_proba)
        return significance
    
    def proba_to_sigma(self,proba):
        """
        Conversion from probabilty to Gaussian standard deviations.
        
        Parameters
        ----------
        proba: float
            Probability.
        
        Returns
        -------
        sigma: float
            Gaussian standard deviations.
        
        """
        sigma=norm.isf(0.5*proba)
        #norm.ppf(1-proba * 0.5 )
        return sigma
    
