"""
Author: Chaya D. Stern
Contact: chaya.stern@choderalab.org
Date: 2016

This code is taken from Grosse and Duvenaud (Testing MCMC code)
"""

import numpy as np
from scipy.special import (psi, polygamma, gammaln)
from scipy.stats import (dirichlet, invgamma, binom)

class GaussianDistribution:
    def __init__(self, mu, sigma_sq):
        self.mu = mu
        self.sigma_sq = sigma_sq

    def log_p(self, x):
        return -0.5 * np.log(2*np.pi) + -0.5*np.log(self.sigma_sq) + -0.5*(x-self.mu) **2 /self.sigma_sq

    def sample(self):
        return np.random.normal(self.mu, np.sqrt(self.sigma_sq))

class DirichletDistribution:
    def __init__(self, alpha):
        self.alpha = alpha

    def log_p(self, x):
        return dirichlet.logpdf(x=x, alpha=self.alpha)

    def sample(self):
        return np.random.dirichlet(self.alpha)


class InverseGamma:
    def __init__(self, a_mu, b_mu):
        self.a_mu = a_mu
        self.b_mu = b_mu

    def log_p(self, x):
        return invgamma.logpdf(x=x, a=self.a_mu, scale=self.b_mu)

    # def sample(self):
    #     return np.random.g
